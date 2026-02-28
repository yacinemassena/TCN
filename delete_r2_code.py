"""Delete tcn_pretrain/ folder from R2 bucket."""
import boto3

# R2 Configuration
R2_ENDPOINT = "https://2a139e9393f803634546ad9d541d37b9.r2.cloudflarestorage.com"
ACCESS_KEY = "fdfa18bf64b18c61bbee64fda98ca20b"
SECRET_KEY = "394c88a7aaf0027feabe74ae20da9b2f743ab861336518a09972bc39534596d8"
BUCKET_NAME = "europe"

# Initialize S3 client for R2
s3 = boto3.client(
    's3',
    endpoint_url=R2_ENDPOINT,
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
    region_name='auto'
)

def delete_folder(prefix):
    """Delete all objects with given prefix."""
    print(f"Deleting all files in {prefix}...")
    
    # List all objects with prefix
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=BUCKET_NAME, Prefix=prefix)
    
    deleted_count = 0
    for page in pages:
        if 'Contents' in page:
            for obj in page['Contents']:
                s3.delete_object(Bucket=BUCKET_NAME, Key=obj['Key'])
                print(f"  Deleted: {obj['Key']}")
                deleted_count += 1
    
    print(f"\n✓ Deleted {deleted_count} files from {prefix}")

if __name__ == "__main__":
    delete_folder("tcn_pretrain/")
