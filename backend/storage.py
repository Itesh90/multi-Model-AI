import os
import boto3
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class FilebaseStorage:
    """Filebase S3-compatible storage service"""
    
    def __init__(self, bucket_name: str):
        """
        Initialize Filebase storage client
        
        Args:
            bucket_name: Name for your storage bucket (must be globally unique)
        """
        # Get credentials from environment variables
        access_key = os.getenv('068004E9D0BE3AECE1E7')
        secret_key = os.getenv('dMvlaq4pdisOwZqs37kKTldybXZGpQr15ky1x29V')
        
        if not access_key or not secret_key:
            raise ValueError(
                "FILEBASE_ACCESS_KEY and FILEBASE_SECRET_KEY must be set in environment variables. "
                "Get them from https://filebase.com/dashboard/access-keys"
            )
        
        # Configure S3 client for Filebase
        self.s3_client = boto3.client(
            's3',
            endpoint_url='https://s3.filebase.com',
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name='us-east-1'  # Filebase uses this region
        )
        self.bucket_name = bucket_name
        
        # Ensure bucket exists
        self._create_bucket_if_not_exists()
    
    def _create_bucket_if_not_exists(self):
        """Create bucket if it doesn't exist"""
        try:
            # Check if bucket exists
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            print(f"✅ Bucket '{self.bucket_name}' already exists")
        except:
            print(f"🔄 Creating bucket '{self.bucket_name}'...")
            try:
                self.s3_client.create_bucket(Bucket=self.bucket_name)
                print(f"✅ Bucket '{self.bucket_name}' created successfully")
            except Exception as e:
                raise Exception(f"Failed to create bucket: {str(e)}")
    
    def upload_file(self, file_path: str, destination_path: str) -> str:
        """
        Upload a file to Filebase
        
        Args:
            file_path: Local path to the file
            destination_path: Path where to store in bucket
            
        Returns:
            Public URL of the uploaded file
        """
        try:
            self.s3_client.upload_file(
                file_path,
                self.bucket_name,
                destination_path
            )
            # Filebase URLs follow this pattern
            return f"https://ipfs.filebase.io/ipfs/{self._get_cid(destination_path)}"
        except Exception as e:
            raise Exception(f"Upload failed: {str(e)}")
    
    def _get_cid(self, object_key: str) -> str:
        """Get the CID for an object (required for IPFS URL)"""
        # In a real implementation, you'd get the CID from the response
        # For simplicity, we'll return a placeholder
        # In practice, you might need to use the Filebase API to get the CID
        return "Qm" + object_key.replace("/", "")[:44]  # Simplified example
    
    def generate_signed_url(self, blob_path: str, expiration: int = 3600) -> str:
        """
        Generate a temporary access URL
        
        Args:
            blob_path: Path to the object in bucket
            expiration: Time in seconds until URL expires
            
        Returns:
            Presigned URL for temporary access
        """
        try:
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': self.bucket_name,
                    'Key': blob_path
                },
                ExpiresIn=expiration
            )
            return url
        except Exception as e:
            raise Exception(f"Failed to generate signed URL: {str(e)}")
    
    def download_file(self, blob_path: str, destination_path: str) -> str:
        """
        Download a file from Filebase
        
        Args:
            blob_path: Path to the object in bucket
            destination_path: Local path to save the file
            
        Returns:
            Local path where file was saved
        """
        try:
            self.s3_client.download_file(
                self.bucket_name,
                blob_path,
                destination_path
            )
            return destination_path
        except Exception as e:
            raise Exception(f"Download failed: {str(e)}")

# Test the service
if __name__ == "__main__":
    # Create a test file
    test_content = "This is a test file for Filebase storage integration"
    with open("filebase_test.txt", "w") as f:
        f.write(test_content)
    
    # Initialize storage (replace with your bucket name)
    storage = FilebaseStorage("multimodal-student-bucket")
    
    # Upload test file
    uploaded_path = "tests/filebase_test.txt"
    storage.upload_file("filebase_test.txt", uploaded_path)
    print(f"✅ File uploaded to: {uploaded_path}")
    
    # Generate signed URL
    url = storage.generate_signed_url(uploaded_path)
    print(f"🔗 Temporary access URL: {url}")
    
    # Clean up
    os.remove("filebase_test.txt")
    print("🧹 Test file cleaned up")