import boto3
import logging
import os

# Set up logging
logger = logging.getLogger(__name__)

class s3:
    @staticmethod
    def get_s3_client():
        """
        Get a boto3 S3 client using environment variables for credentials
        """
        try:
            aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
            aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
            region_name = os.getenv('AWS_REGION', 'us-east-1')
            
            if aws_access_key_id and aws_secret_access_key:
                return boto3.client(
                    's3',
                    aws_access_key_id=aws_access_key_id,
                    aws_secret_access_key=aws_secret_access_key,
                    region_name=region_name
                )
            else:
                logger.warning("AWS credentials not found in environment variables")
                return boto3.client('s3', region_name=region_name)
        except Exception as e:
            logger.error(f"Error creating S3 client: {e}")
            raise 