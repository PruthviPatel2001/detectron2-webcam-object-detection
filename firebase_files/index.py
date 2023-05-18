from datetime import timedelta
import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage
from firebase_admin import firestore
from datetime import datetime
import geocoder
from firebase_admin import db

# datetime object containing current date and time


# dd/mm/YY H:M:S


cred = credentials.Certificate(
    "/Users/pruthvipatel/Desktop/StrandAid_Object_Detection/firebase_files/strandaid-16e48-firebase-adminsdk-1dltc-e05e292a85.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'strandaid-16e48.appspot.com',
    'databaseURL': 'https://strandaid-16e48-default-rtdb.firebaseio.com/'
})

# realtime_db = firebase_admin.initialize_app(cred, {

# })


def addDataToFireBase(image_url, label):
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print("date and time =", dt_string)

    g = geocoder.ip('me')

    print("location:", g.latlng)
    latlng = g.latlng if g.latlng else [None, None]
    data = {
        "image_url": image_url,
        "label": label,
        "time": dt_string,
        "latitude": latlng[0],
        "longitude": latlng[1]
    }

    try:
        print("kook here", data)
        db.reference('drone-capture-data').push().set(data)
    except Exception as e:
        print("see here:", e)


def upload_to_firebase(filename, image_bytes, labels):

    try:

        bucket = storage.bucket()
        blob = bucket.blob(filename)
        blob.upload_from_string(image_bytes, content_type='image/jpeg')
        print("Image uploaded to Firebase Storage successfully!")

        url = blob.generate_signed_url(
            expiration=timedelta(days=7), method='GET')
        addDataToFireBase(url, labels)
        print("URL:", url)
    except Exception as e:
        print("Error in functions uploading image to Firebase Storage:", e)
