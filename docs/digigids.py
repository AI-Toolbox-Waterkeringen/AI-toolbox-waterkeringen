import os
import string
from azure.storage.blob import BlobServiceClient

if __name__ == "__main__":
    # Get connection info from environment variables
    connect_str = os.environ.get("AZURE_SAS")
    container_name = os.environ.get("AZURE_CONTAINER")

    # Initialize markdown table
    digigids_markdown = []
    digigids_markdown.append("| Item              |")
    digigids_markdown.append("| :---------------- |")

    try:
        # List the blobs in the container
        blob_service_client = BlobServiceClient(account_url=connect_str)
        container_client = blob_service_client.get_container_client(container_name)
        blob_list = container_client.list_blobs()
        for blob in blob_list:
            digigids_markdown.append(f"| {blob.name} |")
    except Exception:
        digigids_markdown.append("| {Error connecting to storage...} |")

    # Read original file line by line
    with open("docs/digigids.qmd", "r") as fid:
        src = string.Template(fid.read())

    # Replace placeholder text and write file
    with open("docs/digigids.qmd", "w") as fid:
        digigids_markdown = "\n".join(digigids_markdown)
        dst = src.substitute(placeholder=digigids_markdown)
        fid.write(dst)
