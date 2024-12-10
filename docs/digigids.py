import os
import string
from azure.storage.blob import BlobServiceClient, ContainerClient, BlobPrefix


def list_blobs_hierarchical(
    container_client: ContainerClient,
    folder_entries,
    indent="    ",
    delimiter="/",
    prefix="",
    depth=0,
):
    for blob in container_client.walk_blobs(
        name_starts_with=prefix, delimiter=delimiter
    ):
        if isinstance(blob, BlobPrefix):
            if depth <= 1:
                # Indentation is only added to show nesting in the output
                folder_entries.append(f"| {indent * depth}{blob.name} |")
                depth += 1
                depth, folder_entries = list_blobs_hierarchical(
                    container_client,
                    folder_entries,
                    indent=indent,
                    delimiter=delimiter,
                    prefix=blob.name,
                    depth=depth,
                )
                depth -= 1
            else:
                break
        else:
            continue

    return depth, folder_entries


if __name__ == "__main__":
    # Get connection info from environment variables
    connect_str = os.environ.get("AZURE_SAS")
    container_name = os.environ.get("AZURE_CONTAINER")

    # Initialize markdown table
    digigids_markdown = []
    digigids_markdown.append("| Naam              |")
    digigids_markdown.append("| :---------------- |")

    try:
        # List the blobs in the container
        blob_service_client = BlobServiceClient(account_url=connect_str)
        container_client = blob_service_client.get_container_client(container_name)
        _, folder_entries = list_blobs_hierarchical(container_client, [])
        digigids_markdown += folder_entries
    except Exception:
        digigids_markdown.append("| Error connecting to storage... |")

    # Read original file line by line
    with open("docs/digigids.qmd", "r") as fid:
        src = string.Template(fid.read())

    # Replace placeholder text and write file
    with open("docs/digigids.qmd", "w") as fid:
        digigids_markdown = "\n".join(digigids_markdown)
        dst = src.substitute(placeholder=digigids_markdown)
        fid.write(dst)
