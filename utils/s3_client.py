"""
S3 client utilities for AWS S3 operations.
"""
import io
import pandas as pd
import boto3
from botocore.exceptions import ClientError
from datetime import datetime, timedelta
from config import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION, S3_BUCKET_NAME


def get_s3_client():
    """Get S3 client with credentials from environment variables"""
    return boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION
    )


def upload_to_s3(file_obj, key):
    """Upload a file object to S3"""
    s3 = get_s3_client()
    try:
        s3.upload_fileobj(file_obj, S3_BUCKET_NAME, key)
        return True
    except ClientError as e:
        print(f"[ERROR] S3 upload failed: {e}")
        return False


def read_from_s3(key, as_dataframe=True):
    """Read a file from S3, optionally return as DataFrame"""
    s3 = get_s3_client()
    try:
        response = s3.get_object(Bucket=S3_BUCKET_NAME, Key=key)
        if as_dataframe:
            # Detect file type from key
            if key.endswith('.csv'):
                return pd.read_csv(io.BytesIO(response['Body'].read()))
            elif key.endswith('.xlsx'):
                return pd.read_excel(io.BytesIO(response['Body'].read()), engine='openpyxl')
            elif key.endswith('.xls'):
                return pd.read_excel(io.BytesIO(response['Body'].read()), engine='xlrd')
        else:
            return response['Body'].read()
    except ClientError as e:
        print(f"[ERROR] S3 read failed: {e}")
        return None


def save_dataframe_to_s3(df, key):
    """Save a DataFrame to S3 as CSV"""
    s3 = get_s3_client()
    try:
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        s3.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=key,
            Body=csv_buffer.getvalue()
        )
        return True
    except ClientError as e:
        print(f"[ERROR] S3 dataframe save failed: {e}")
        return False


def delete_from_s3(key):
    """Delete a file from S3"""
    s3 = get_s3_client()
    try:
        s3.delete_object(Bucket=S3_BUCKET_NAME, Key=key)
        return True
    except ClientError as e:
        print(f"[ERROR] S3 delete failed: {e}")
        return False


def generate_presigned_url(key, expiration=3600):
    """Generate a pre-signed URL for a file in S3"""
    s3 = get_s3_client()
    try:
        response = s3.generate_presigned_url(
            'get_object',
            Params={'Bucket': S3_BUCKET_NAME, 'Key': key},
            ExpiresIn=expiration
        )
        return response
    except ClientError as e:
        print(f"[ERROR] Failed to generate presigned URL: {e}")
        return None


def cleanup_old_s3_files():
    """Remove files older than 1 hour from S3 bucket""" 
    s3 = get_s3_client()
    one_hour_ago = datetime.now() - timedelta(hours=1)
    
    try:
        objects = s3.list_objects_v2(Bucket=S3_BUCKET_NAME)
        if 'Contents' in objects:
            for obj in objects['Contents']:
                # If object is older than 1 hour 
                if obj['LastModified'].replace(tzinfo=None) < one_hour_ago:
                    try:
                        s3.delete_object(Bucket=S3_BUCKET_NAME, Key=obj['Key'])
                        print(f"[CLEANUP] Removed old S3 file: {obj['Key']}")
                    except Exception as e:
                        print(f"[ERROR] Failed to remove old S3 file {obj['Key']}: {e}")
    except Exception as e:
        print(f"[ERROR] Error listing S3 objects: {e}")
