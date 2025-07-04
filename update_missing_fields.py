from pymongo import MongoClient

# ✅ Connect to MongoDB
mongo_uri = "mongodb+srv://GlucoPredict:Gluco123@cluster1.3hlg9y3.mongodb.net/diabetes?retryWrites=true&w=majority"
client = MongoClient(mongo_uri)
db = client["diabetes"]
collection = db["predictions"]

# ✅ Update documents missing 'Name' or 'Email'
result_name = collection.update_many(
    {"Name": {"$exists": False}},
    {"$set": {"Name": "Unknown"}}
)

result_email = collection.update_many(
    {"Email": {"$exists": False}},
    {"$set": {"Email": "Unknown"}}
)

print(f"🔄 Name updated in {result_name.modified_count} documents.")
print(f"🔄 Email updated in {result_email.modified_count} documents.")
