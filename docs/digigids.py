import os
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
    lines = []
    with open("docs/digigids.qmd", "r") as fid:
        lines = fid.readlines()

    # Replace placeholder text
    for i, line in enumerate(lines):
        if "[placeholder]" in line:
            lines[i] = line.replace("[placeholder]", "\n".join(digigids_markdown))

    # Write modified file
    with open("docs/digigids.qmd", "w") as fid:
        for line in lines:
            fid.write(f"{line}\n")
