import csv
from pathlib import Path

from firebase_admin import credentials, firestore, initialize_app

CRED_FILE = Path(__file__).resolve().parent / 'key.json'
COLLECTION_NAMES = ['customers, requests']
DIRECTION = firestore.Query.DESCENDING


def main():
    client = firebase_client(str(CRED_FILE))

    for collection_name in COLLECTION_NAMES:
        collection = client.collection(collection_name)

        filename = 'export_{}.csv'.format(collection_name)

        with open(filename, 'w') as file:
            for idx, snapshot in enumerate(collection.get()):
                if idx == 0:
                    fields = snapshot.to_dict().keys()
                    writer = csv_writer(file, fields)
                    writer.writeheader()
                if snapshot.to_dict().keys() != fields:
                    print('Skipping because of different fields: {}'.format(snapshot.id))
                else:
                    data = snapshot.to_dict()
                    writer.writerow(data)



def firebase_client(cred_file):
    """Generate Firebase client"""
    cred = credentials.Certificate(cred_file)
    app = initialize_app(credential=cred)
    client = firestore.client(app=app)
    return client


def csv_writer(file, fields):
    """Generate CSV writer"""
    return csv.DictWriter(file, fields)


if __name__ == '__main__':
    main()
