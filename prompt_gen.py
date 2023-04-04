from constants import orientation_override_words
from word_present import word_present
from constants import negative_prompts, prompt_modifiers


def people_prompt_gen(action, age, sex, clothing):
    orientation_specified = word_present(action, orientation_override_words)
    if not orientation_specified:
        neg_prompt = 'facing the camera, ' + negative_prompts['people']
        prompt_mod = prompt_modifiers['people'] + ', facing away'
    else:
        neg_prompt = negative_prompts['people']
        prompt_mod = prompt_modifiers['people']

    prompt = age + ' ' + sex + ' wearing ' + clothing + ' ' + action + ', ' + prompt_mod
    print('PROMPT:')
    print(prompt)
    print('NEG_PROMPT: ')
    print(neg_prompt)

    return prompt, neg_prompt
