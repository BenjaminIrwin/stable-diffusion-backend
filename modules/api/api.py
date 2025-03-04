import asyncio
import base64
import datetime
import io
import json
import time
import datetime
import uvicorn
import gradio as gr
from threading import Lock
from io import BytesIO
from threading import Lock
from typing import Callable

import firebase_admin
import piexif
import piexif.helper
import sentry_sdk
import uvicorn
from PIL import PngImagePlugin, Image
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, Response
from fastapi.routing import APIRoute
from firebase_admin import credentials, firestore
from google.cloud import firestore as gfirestore
from gradio.processing_utils import decode_base64_to_file
from gradio.processing_utils import decode_base64_to_file
from fastapi import APIRouter, Depends, FastAPI, Request, Response
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.exceptions import HTTPException
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from secrets import compare_digest

import modules.shared as shared
from constants import *
from make_a_hole_in_image import make_a_hole_in_image
from modules import devices
from modules import sd_samplers, deepbooru, sd_hijack, images, scripts, ui, postprocessing
from modules.api.models import *
from modules.hypernetworks.hypernetwork import create_hypernetwork, train_hypernetwork
from modules.processing import StableDiffusionProcessingTxt2Img, StableDiffusionProcessingImg2Img, process_images, \
    RembgProcessing
from modules.realesrgan_model import get_realesrgan_models
from modules.s3 import upload_base64_file, upload_base64_files
from modules.sd_models import checkpoints_list
from PIL import PngImagePlugin,Image
from modules.sd_models import checkpoints_list, unload_model_weights, reload_model_weights
from modules.sd_models_config import find_checkpoint_config_near_filename
from modules.textual_inversion.preprocess import preprocess
from modules.textual_inversion.textual_inversion import create_embedding, train_embedding
from prompt_gen import people_prompt_gen

sentry_sdk.init(
    dsn="https://db21fe40066142b093719b3e9a7d57f6@o4504922099089408.ingest.sentry.io/4504922100465669",
    traces_sample_rate=1.0,
)

# Fetch the service account key JSON file contents
cred = credentials.Certificate(fs_sa_key)
app = firebase_admin.initialize_app(cred)
db = firestore.client()
users_db = db.collection('customers')
requests_db = db.collection('requests')

def upscaler_to_index(name: str):
    try:
        return [x.name.lower() for x in shared.sd_upscalers].index(name.lower())
    except:
        raise HTTPException(status_code=400, detail=f"Invalid upscaler, needs to be one of these: {' , '.join([x.name for x in sd_upscalers])}")

def script_name_to_index(name, scripts):
    try:
        return [script.title().lower() for script in scripts].index(name.lower())
    except:
        raise HTTPException(status_code=422, detail=f"Script '{name}' not found")


def validate_sampler_name(name):
    config = sd_samplers.all_samplers_map.get(name, None)
    if config is None:
        raise HTTPException(status_code=404, detail="Sampler not found")

    return name


def setUpscalers(req: dict):
    reqDict = vars(req)
    reqDict['extras_upscaler_1'] = reqDict.pop('upscaler_1', None)
    reqDict['extras_upscaler_2'] = reqDict.pop('upscaler_2', None)
    return reqDict


def decode_base64_to_image(encoding):
    if encoding.startswith("data:image/"):
        encoding = encoding.split(";")[1].split(",")[1]
    try:
        image = Image.open(BytesIO(base64.b64decode(encoding)))
        return image
    except Exception as err:
        raise HTTPException(status_code=500, detail="Invalid encoded image")


def encode_pil_to_base64(image):
    with io.BytesIO() as output_bytes:

        if opts.samples_format.lower() == 'png':
            use_metadata = False
            metadata = PngImagePlugin.PngInfo()
            for key, value in image.info.items():
                if isinstance(key, str) and isinstance(value, str):
                    metadata.add_text(key, value)
                    use_metadata = True
            image.save(output_bytes, format="PNG", pnginfo=(metadata if use_metadata else None),
                       quality=opts.jpeg_quality)

        elif opts.samples_format.lower() in ("jpg", "jpeg", "webp"):
            parameters = image.info.get('parameters', None)
            exif_bytes = piexif.dump({
                "Exif": {
                    piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump(parameters or "", encoding="unicode")}
            })
            if opts.samples_format.lower() in ("jpg", "jpeg"):
                image.save(output_bytes, format="JPEG", exif=exif_bytes, quality=opts.jpeg_quality)
            else:
                image.save(output_bytes, format="WEBP", exif=exif_bytes, quality=opts.jpeg_quality)

        else:
            raise HTTPException(status_code=500, detail="Invalid image format")

        bytes_data = output_bytes.getvalue()

    return base64.b64encode(bytes_data)


def api_middleware(app: FastAPI):
    rich_available = True
    try:
        import anyio # importing just so it can be placed on silent list
        import starlette # importing just so it can be placed on silent list
        from rich.console import Console
        console = Console()
    except:
        import traceback
        rich_available = False

    @app.middleware("http")
    async def log_and_time(req: Request, call_next):
        ts = time.time()
        res: Response = await call_next(req)
        duration = str(round(time.time() - ts, 4))
        res.headers["X-Process-Time"] = duration
        endpoint = req.scope.get('path', 'err')
        if shared.cmd_opts.api_log and endpoint.startswith('/sdapi'):
            print('API {t} {code} {prot}/{ver} {method} {endpoint} {cli} {duration}'.format(
                t=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
                code=res.status_code,
                ver=req.scope.get('http_version', '0.0'),
                cli=req.scope.get('client', ('0:0.0.0', 0))[0],
                prot=req.scope.get('scheme', 'err'),
                method=req.scope.get('method', 'err'),
                endpoint=endpoint,
                duration=duration,
            ))
        return res


class AuthenticationRouter(APIRoute):
    def get_route_handler(self) -> Callable:
        original_route_handler = super().get_route_handler()

        def log_increment_generation_count(id, request_body, task):
            if task == "rembg":
                amount = len(request_body['init_images'])
            else:
                amount = request_body['batch_size'] * request_body['n_iter']
            user_ref = users_db.document(id)
            user_ref.update({"cur_generations": gfirestore.Increment(amount)})

        def auth(request):
            # check if api_key is in headers

            if 'api_key' in request.headers:
                api_key = request.headers['api_key']
                # Query firebase firestore database with api_key
                res = users_db.where('api_key', '==', api_key).get()
                if len(res) > 0:
                    user = res[0]
                    # if user.get('cur_generations') >= user.get('generation_limit'):
                    #     raise HTTPException(status_code=429, detail="Max generations reached.")
                    return user.id
                else:
                    raise HTTPException(status_code=403, detail="Incorrect api_key provided.")
            else:
                raise HTTPException(status_code=403, detail="No api_key provided.")

        async def log(request: Request, response: Response, user_id):
            task = request.scope.get('path', 'err').split('/')[-1]
            request_body = await request.json()
            log_increment_generation_count(user_id, request_body, task)
            response_body = json.loads(response.body)
            log_images(user_id, task, request_body, response_body)

        def log_images(user_id, task, request_body, response_body):
            init_images = upload_base64_files(request_body['init_images'])
            output_images = upload_base64_files(response_body['images'])
            data = {
                'user_id': user_id,
                'task': task,
                'init_images': init_images,
                'output_images': output_images,
                'timestamp': gfirestore.SERVER_TIMESTAMP
            }
            # If the request has a mask field which is not null
            if 'mask' in request_body and request_body['mask']:
                mask = upload_base64_file(request_body['mask'])
                data['mask'] = mask
            if task != 'rembg':
                data['params'] = response_body['parameters']



            requests_db.add(data)

        async def custom_route_handler(request: Request) -> Response:
            before = time.time()
            path = request.scope.get('path', 'err')
            if path.startswith('/sdapi'):
                user_id = auth(request)
            response: Response = await original_route_handler(request)
            duration = time.time() - before
            response.headers["X-Response-Time"] = str(duration)
            # if 200 and path begins with 'sdapi'
            if response.status_code == 200 and path.startswith('/sdapi') and not path.endswith('test'):
                asyncio.ensure_future(log(request, response, user_id))
            return response

        return custom_route_handler


class Api:
    def __init__(self, app: FastAPI, queue_lock: Lock):
        self.router = APIRouter(route_class=AuthenticationRouter)
        self.app = app
        self.queue_lock = queue_lock
        api_middleware(self.app)

        # add api route for root which always returns 200
        # self.router.add_api_route("/", lambda: Response(status_code=200))

        self.add_api_route_auth("/sdapi/v1/person", self.img2imgapi, methods=["POST"], response_model=ImageToImageResponse)
        self.add_api_route_auth("/sdapi/v1/person/test", self.img2imgapitest, methods=["POST"], response_model=ImageToImageResponse)

        self.add_api_route_auth("/sdapi/v1/rembg", self.rembgapi, methods=["POST"], response_model=ImageToImageResponse)
        self.add_api_route_auth("/sdapi/v1/rembg/test", self.rembgapitest, methods=["POST"], response_model=ImageToImageResponse)

        # Return 200 for root
        self.router.add_api_route("/", self.healthcheckapi, methods=["GET"])

        self.default_script_arg_txt2img = []
        self.default_script_arg_img2img = []

    def add_api_route_auth(self, path: str, endpoint, **kwargs):
        return self.router.add_api_route(path, endpoint, route_class_override=AuthenticationRouter, **kwargs)

    def get_selectable_script(self, script_name, script_runner):
        if script_name is None or script_name == "":
            return None, None

        script_idx = script_name_to_index(script_name, script_runner.selectable_scripts)
        script = script_runner.selectable_scripts[script_idx]
        return script, script_idx

    def get_scripts_list(self):
        t2ilist = [str(title.lower()) for title in scripts.scripts_txt2img.titles]
        i2ilist = [str(title.lower()) for title in scripts.scripts_img2img.titles]

        return ScriptsList(txt2img = t2ilist, img2img = i2ilist)

    def get_script(self, script_name, script_runner):
        if script_name is None or script_name == "":
            return None, None

        script_idx = script_name_to_index(script_name, script_runner.scripts)
        return script_runner.scripts[script_idx]

    def init_default_script_args(self, script_runner):
        #find max idx from the scripts in runner and generate a none array to init script_args
        last_arg_index = 1
        for script in script_runner.scripts:
            if last_arg_index < script.args_to:
                last_arg_index = script.args_to
        # None everywhere except position 0 to initialize script args
        script_args = [None]*last_arg_index
        script_args[0] = 0

        # get default values
        with gr.Blocks(): # will throw errors calling ui function without this
            for script in script_runner.scripts:
                if script.ui(script.is_img2img):
                    ui_default_values = []
                    for elem in script.ui(script.is_img2img):
                        ui_default_values.append(elem.value)
                    script_args[script.args_from:script.args_to] = ui_default_values
        return script_args

    def init_script_args(self, request, default_script_args, selectable_scripts, selectable_idx, script_runner):
        script_args = default_script_args.copy()
        # position 0 in script_arg is the idx+1 of the selectable script that is going to be run when using scripts.scripts_*2img.run()
        if selectable_scripts:
            script_args[selectable_scripts.args_from:selectable_scripts.args_to] = request.script_args
            script_args[0] = selectable_idx + 1

        # Now check for always on scripts
        if request.alwayson_scripts and (len(request.alwayson_scripts) > 0):
            for alwayson_script_name in request.alwayson_scripts.keys():
                alwayson_script = self.get_script(alwayson_script_name, script_runner)
                if alwayson_script == None:
                    raise HTTPException(status_code=422, detail=f"always on script {alwayson_script_name} not found")
                # Selectable script in always on script param check
                if alwayson_script.alwayson == False:
                    raise HTTPException(status_code=422, detail=f"Cannot have a selectable script in the always on scripts params")
                # always on script with no arg should always run so you don't really need to add them to the requests
                if "args" in request.alwayson_scripts[alwayson_script_name]:
                    script_args[alwayson_script.args_from:alwayson_script.args_to] = request.alwayson_scripts[alwayson_script_name]["args"]
        return script_args

    def text2imgapi(self, txt2imgreq: StableDiffusionTxt2ImgProcessingAPI):
        script_runner = scripts.scripts_txt2img
        if not script_runner.scripts:
            script_runner.initialize_scripts(False)
            ui.create_ui()

        script_idx = script_name_to_index(script_name, script_runner.selectable_scripts)
        script = script_runner.selectable_scripts[script_idx]
        return script, script_idx

    def healthcheckapi(self):
        return

    def text2imgapi(self, txt2imgreq: StableDiffusionTxt2ImgProcessingAPI):
        script, script_idx = self.get_script(txt2imgreq.script_name, scripts.scripts_txt2img)
        if not self.default_script_arg_txt2img:
            self.default_script_arg_txt2img = self.init_default_script_args(script_runner)
        selectable_scripts, selectable_script_idx = self.get_selectable_script(txt2imgreq.script_name, script_runner)

        populate = txt2imgreq.copy(update={  # Override __init__ params
            "sampler_name": validate_sampler_name(txt2imgreq.sampler_name or txt2imgreq.sampler_index),
            "do_not_save_samples": not txt2imgreq.save_images,
            "do_not_save_grid": not txt2imgreq.save_images,
        })
        if populate.sampler_name:
            populate.sampler_index = None  # prevent a warning later on

        args = vars(populate)
        args.pop('script_name', None)
        args.pop('script_args', None) # will refeed them to the pipeline directly after initializing them
        args.pop('alwayson_scripts', None)

        script_args = self.init_script_args(txt2imgreq, self.default_script_arg_txt2img, selectable_scripts, selectable_script_idx, script_runner)

        send_images = args.pop('send_images', True)
        args.pop('save_images', None)

        with self.queue_lock:
            p = StableDiffusionProcessingTxt2Img(sd_model=shared.sd_model, **args)
            p.scripts = script_runner
            p.outpath_grids = opts.outdir_txt2img_grids
            p.outpath_samples = opts.outdir_txt2img_samples

            shared.state.begin()
            if selectable_scripts != None:
                p.script_args = script_args
                processed = scripts.scripts_txt2img.run(p, *p.script_args) # Need to pass args as list here
            else:
                p.script_args = tuple(script_args) # Need to pass args as tuple here
                processed = process_images(p)
            shared.state.end()

        b64images = list(map(encode_pil_to_base64, processed.images)) if send_images else []

        return TextToImageResponse(images=b64images, parameters=vars(txt2imgreq), info=processed.js())

    def rembgapitest(self, rembgreq: RembgProcessingAPI):
        init_images = rembgreq.init_images
        if init_images is None:
            raise HTTPException(status_code=404, detail="Init image not found")


        processed = []
        with self.queue_lock:
            for img in init_images:
                output_image = make_a_hole_in_image(img)
                processed.append(output_image)

        output_images = list(map(encode_pil_to_base64, processed))

        return ImageToImageResponse(images=output_images, parameters=vars(rembgreq), info="rembg")

    def rembgapi(self, rembgreq: RembgProcessingAPI):
        init_images = rembgreq.init_images
        if init_images is None:
            raise HTTPException(status_code=404, detail="Init image not found")

        with self.queue_lock:
            p = RembgProcessing(init_images)
            processed = p.process()

        output_images = list(map(encode_pil_to_base64, processed))

        return ImageToImageResponse(images=output_images, parameters=vars(rembgreq), info="rembg")

    def img2imgapitest(self, img2imgreq: StableDiffusionImg2ImgProcessingAPI):

        # Load the test image
        test_image = Image.open("test.jpeg")

        # Convert to base64 string without the header
        test_image = encode_pil_to_base64(test_image)

        test_images = [test_image, test_image, test_image, test_image]

        return ImageToImageResponse(images=test_images, parameters=vars(img2imgreq), info="test")

    def img2imgapi(self, img2imgreq: StableDiffusionImg2ImgProcessingAPI):

        init_images = img2imgreq.init_images
        prompt, negative_prompt = people_prompt_gen(img2imgreq.action, img2imgreq.age, img2imgreq.sex, img2imgreq.clothing)
        img2imgreq.negative_prompt = negative_prompt
        img2imgreq.prompt = prompt
        if init_images is None:
            raise HTTPException(status_code=404, detail="Init image not found")

        mask = img2imgreq.mask
        if mask:
            mask = decode_base64_to_image(mask)

        script_runner = scripts.scripts_img2img
        if not script_runner.scripts:
            script_runner.initialize_scripts(True)
            ui.create_ui()
        if not self.default_script_arg_img2img:
            self.default_script_arg_img2img = self.init_default_script_args(script_runner)
        selectable_scripts, selectable_script_idx = self.get_selectable_script(img2imgreq.script_name, script_runner)

        populate = img2imgreq.copy(update={  # Override __init__ params
            "sampler_name": validate_sampler_name(img2imgreq.sampler_name or img2imgreq.sampler_index),
            "do_not_save_samples": not img2imgreq.save_images,
            "do_not_save_grid": not img2imgreq.save_images,
            "mask": mask,
        })

        if populate.sampler_name:
            populate.sampler_index = None  # prevent a warning later on

        args = vars(populate)
        args.pop('include_init_images',
                 None)  # this is meant to be done by "exclude": True in model, but it's for a reason that I cannot determine.
        args.pop('script_name', None)
        args.pop('script_args', None)  # will refeed them to the pipeline directly after initializing them
        args.pop('alwayson_scripts', None)

        # script_args = self.init_script_args(img2imgreq, self.default_script_arg_img2img, selectable_scripts, selectable_script_idx, script_runner)

        script_args = [0, '<ul>\n<li><code>CFG Scale</code> should be 2 or lower.</li>\n</ul>\n', True, True, '', '', True, 50, True, 1, 0, False, 4, 1, 'None', '<p style="margin-bottom:0.75em">Recommended settings: Sampling Steps: 80-100, Sampler: Euler a, Denoising strength: 0.8</p>', 128, 8, ['left', 'right', 'up', 'down'], 1, 0.05, 128, 4, 0, ['left', 'right', 'up', 'down'], False, False, 'positive', 'comma', 0, False, False, '', '<p style="margin-bottom:0.75em">Will upscale the image by the selected scale factor; use width and height sliders to set tile size</p>', 64, 0, 2, 1, '', 0, '', 0, '', True, False, False, False, 0]
        print("script_args", script_args)
        print(type(script_args))

        send_images = args.pop('send_images', True)
        args.pop('save_images', None)

        with self.queue_lock:
            p = StableDiffusionProcessingImg2Img(sd_model=shared.sd_model, **args)
            p.init_images = [decode_base64_to_image(x) for x in init_images]
            # Make all init_images RGB
            p.init_images = [x.convert('RGB') for x in p.init_images]
            # Make mask rgb
            if p.mask:
                p.mask = p.mask.convert('RGB')
            p.scripts = script_runner
            p.outpath_grids = opts.outdir_img2img_grids
            p.outpath_samples = opts.outdir_img2img_samples

            shared.state.begin()
            if selectable_scripts != None:
                p.script_args = script_args
                processed = scripts.scripts_img2img.run(p, *p.script_args) # Need to pass args as list here
            else:
                p.script_args = tuple(script_args) # Need to pass args as tuple here
                processed = process_images(p)
            shared.state.end()

        b64images = list(map(encode_pil_to_base64, processed.images)) if send_images else []

        if not img2imgreq.include_init_images:
            img2imgreq.init_images = None
            img2imgreq.mask = None

        return ImageToImageResponse(images=b64images, parameters=vars(img2imgreq), info=processed.js())

    def extras_single_image_api(self, req: ExtrasSingleImageRequest):
        reqDict = setUpscalers(req)

        reqDict['image'] = decode_base64_to_image(reqDict['image'])

        with self.queue_lock:
            result = postprocessing.run_extras(extras_mode=0, image_folder="", input_dir="", output_dir="", save_output=False, **reqDict)

        return ExtrasSingleImageResponse(image=encode_pil_to_base64(result[0][0]), html_info=result[1])

    def extras_batch_images_api(self, req: ExtrasBatchImagesRequest):
        reqDict = setUpscalers(req)

        def prepareFiles(file):
            file = decode_base64_to_file(file.data, file_path=file.name)
            file.orig_name = file.name
            return file

        reqDict['image_folder'] = list(map(prepareFiles, reqDict['imageList']))
        reqDict.pop('imageList')

        with self.queue_lock:
            result = postprocessing.run_extras(extras_mode=1, image="", input_dir="", output_dir="", save_output=False, **reqDict)

        return ExtrasBatchImagesResponse(images=list(map(encode_pil_to_base64, result[0])), html_info=result[1])

    def pnginfoapi(self, req: PNGInfoRequest):
        if(not req.image.strip()):
            return PNGInfoResponse(info="")

        image = decode_base64_to_image(req.image.strip())
        if image is None:
            return PNGInfoResponse(info="")

        geninfo, items = images.read_info_from_image(image)
        if geninfo is None:
            geninfo = ""

        items = {**{'parameters': geninfo}, **items}

        return PNGInfoResponse(info=geninfo, items=items)

    def progressapi(self, req: ProgressRequest = Depends()):
        # copy from check_progress_call of ui.py

        if shared.state.job_count == 0:
            return ProgressResponse(progress=0, eta_relative=0, state=shared.state.dict(), textinfo=shared.state.textinfo)

        # avoid dividing zero
        progress = 0.01

        if shared.state.job_count > 0:
            progress += shared.state.job_no / shared.state.job_count
        if shared.state.sampling_steps > 0:
            progress += 1 / shared.state.job_count * shared.state.sampling_step / shared.state.sampling_steps

        time_since_start = time.time() - shared.state.time_start
        eta = (time_since_start/progress)
        eta_relative = eta-time_since_start

        progress = min(progress, 1)

        shared.state.set_current_image()

        current_image = None
        if shared.state.current_image and not req.skip_current_image:
            current_image = encode_pil_to_base64(shared.state.current_image)

        return ProgressResponse(progress=progress, eta_relative=eta_relative, state=shared.state.dict(), current_image=current_image, textinfo=shared.state.textinfo)

    def interrogateapi(self, interrogatereq: InterrogateRequest):
        image_b64 = interrogatereq.image
        if image_b64 is None:
            raise HTTPException(status_code=404, detail="Image not found")

        img = decode_base64_to_image(image_b64)
        img = img.convert('RGB')

        # Override object param
        with self.queue_lock:
            if interrogatereq.model == "clip":
                processed = shared.interrogator.interrogate(img)
            elif interrogatereq.model == "deepdanbooru":
                processed = deepbooru.model.tag(img)
            else:
                raise HTTPException(status_code=404, detail="Model not found")

        return InterrogateResponse(caption=processed)

    def interruptapi(self):
        shared.state.interrupt()

        return {}

    def unloadapi(self):
        unload_model_weights()

        return {}

    def reloadapi(self):
        reload_model_weights()

        return {}

    def skip(self):
        shared.state.skip()

    def get_config(self):
        options = {}
        for key in shared.opts.data.keys():
            metadata = shared.opts.data_labels.get(key)
            if(metadata is not None):
                options.update({key: shared.opts.data.get(key, shared.opts.data_labels.get(key).default)})
            else:
                options.update({key: shared.opts.data.get(key, None)})

        return options

    def set_config(self, req: Dict[str, Any]):
        for k, v in req.items():
            shared.opts.set(k, v)

        shared.opts.save(shared.config_filename)
        return

    def get_cmd_flags(self):
        return vars(shared.cmd_opts)

    def get_samplers(self):
        return [{"name": sampler[0], "aliases":sampler[2], "options":sampler[3]} for sampler in sd_samplers.all_samplers]

    def get_upscalers(self):
        return [
            {
                "name": upscaler.name,
                "model_name": upscaler.scaler.model_name,
                "model_path": upscaler.data_path,
                "model_url": None,
                "scale": upscaler.scale,
            }
            for upscaler in shared.sd_upscalers
        ]

    def get_sd_models(self):
        return [{"title": x.title, "model_name": x.model_name, "hash": x.shorthash, "sha256": x.sha256, "filename": x.filename, "config": find_checkpoint_config_near_filename(x)} for x in checkpoints_list.values()]

    def get_hypernetworks(self):
        return [{"name": name, "path": shared.hypernetworks[name]} for name in shared.hypernetworks]

    def get_face_restorers(self):
        return [{"name":x.name(), "cmd_dir": getattr(x, "cmd_dir", None)} for x in shared.face_restorers]

    def get_realesrgan_models(self):
        return [{"name":x.name,"path":x.data_path, "scale":x.scale} for x in get_realesrgan_models(None)]

    def get_prompt_styles(self):
        styleList = []
        for k in shared.prompt_styles.styles:
            style = shared.prompt_styles.styles[k]
            styleList.append({"name":style[0], "prompt": style[1], "negative_prompt": style[2]})

        return styleList

    def get_embeddings(self):
        db = sd_hijack.model_hijack.embedding_db

        def convert_embedding(embedding):
            return {
                "step": embedding.step,
                "sd_checkpoint": embedding.sd_checkpoint,
                "sd_checkpoint_name": embedding.sd_checkpoint_name,
                "shape": embedding.shape,
                "vectors": embedding.vectors,
            }

        def convert_embeddings(embeddings):
            return {embedding.name: convert_embedding(embedding) for embedding in embeddings.values()}

        return {
            "loaded": convert_embeddings(db.word_embeddings),
            "skipped": convert_embeddings(db.skipped_embeddings),
        }

    def refresh_checkpoints(self):
        shared.refresh_checkpoints()

    def create_embedding(self, args: dict):
        try:
            shared.state.begin()
            filename = create_embedding(**args) # create empty embedding
            sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings() # reload embeddings so new one can be immediately used
            shared.state.end()
            return CreateResponse(info = "create embedding filename: {filename}".format(filename = filename))
        except AssertionError as e:
            shared.state.end()
            return TrainResponse(info = "create embedding error: {error}".format(error = e))

    def create_hypernetwork(self, args: dict):
        try:
            shared.state.begin()
            filename = create_hypernetwork(**args) # create empty embedding
            shared.state.end()
            return CreateResponse(info = "create hypernetwork filename: {filename}".format(filename = filename))
        except AssertionError as e:
            shared.state.end()
            return TrainResponse(info = "create hypernetwork error: {error}".format(error = e))

    def preprocess(self, args: dict):
        try:
            shared.state.begin()
            preprocess(**args) # quick operation unless blip/booru interrogation is enabled
            shared.state.end()
            return PreprocessResponse(info = 'preprocess complete')
        except KeyError as e:
            shared.state.end()
            return PreprocessResponse(info = "preprocess error: invalid token: {error}".format(error = e))
        except AssertionError as e:
            shared.state.end()
            return PreprocessResponse(info = "preprocess error: {error}".format(error = e))
        except FileNotFoundError as e:
            shared.state.end()
            return PreprocessResponse(info = 'preprocess error: {error}'.format(error = e))

    def train_embedding(self, args: dict):
        try:
            shared.state.begin()
            apply_optimizations = shared.opts.training_xattention_optimizations
            error = None
            filename = ''
            if not apply_optimizations:
                sd_hijack.undo_optimizations()
            try:
                embedding, filename = train_embedding(**args) # can take a long time to complete
            except Exception as e:
                error = e
            finally:
                if not apply_optimizations:
                    sd_hijack.apply_optimizations()
                shared.state.end()
            return TrainResponse(info = "train embedding complete: filename: {filename} error: {error}".format(filename = filename, error = error))
        except AssertionError as msg:
            shared.state.end()
            return TrainResponse(info = "train embedding error: {msg}".format(msg = msg))

    def train_hypernetwork(self, args: dict):
        try:
            shared.state.begin()
            shared.loaded_hypernetworks = []
            apply_optimizations = shared.opts.training_xattention_optimizations
            error = None
            filename = ''
            if not apply_optimizations:
                sd_hijack.undo_optimizations()
            try:
                hypernetwork, filename = train_hypernetwork(**args)
            except Exception as e:
                error = e
            finally:
                shared.sd_model.cond_stage_model.to(devices.device)
                shared.sd_model.first_stage_model.to(devices.device)
                if not apply_optimizations:
                    sd_hijack.apply_optimizations()
                shared.state.end()
            return TrainResponse(info="train embedding complete: filename: {filename} error: {error}".format(filename=filename, error=error))
        except AssertionError as msg:
            shared.state.end()
            return TrainResponse(info="train embedding error: {error}".format(error=error))

    def get_memory(self):
        try:
            import os, psutil
            process = psutil.Process(os.getpid())
            res = process.memory_info() # only rss is cross-platform guaranteed so we dont rely on other values
            ram_total = 100 * res.rss / process.memory_percent() # and total memory is calculated as actual value is not cross-platform safe
            ram = { 'free': ram_total - res.rss, 'used': res.rss, 'total': ram_total }
        except Exception as err:
            ram = { 'error': f'{err}' }
        try:
            import torch
            if torch.cuda.is_available():
                s = torch.cuda.mem_get_info()
                system = { 'free': s[0], 'used': s[1] - s[0], 'total': s[1] }
                s = dict(torch.cuda.memory_stats(shared.device))
                allocated = { 'current': s['allocated_bytes.all.current'], 'peak': s['allocated_bytes.all.peak'] }
                reserved = { 'current': s['reserved_bytes.all.current'], 'peak': s['reserved_bytes.all.peak'] }
                active = { 'current': s['active_bytes.all.current'], 'peak': s['active_bytes.all.peak'] }
                inactive = { 'current': s['inactive_split_bytes.all.current'], 'peak': s['inactive_split_bytes.all.peak'] }
                warnings = { 'retries': s['num_alloc_retries'], 'oom': s['num_ooms'] }
                cuda = {
                    'system': system,
                    'active': active,
                    'allocated': allocated,
                    'reserved': reserved,
                    'inactive': inactive,
                    'events': warnings,
                }
            else:
                cuda = { 'error': 'unavailable' }
        except Exception as err:
            cuda = { 'error': f'{err}' }
        return MemoryResponse(ram = ram, cuda = cuda)

    def launch(self, server_name, port):
        self.app.include_router(self.router)
        print('Launching server on {server_name}:{port}'.format(server_name=server_name, port=port))
        uvicorn.run(self.app, host=server_name, port=port)
