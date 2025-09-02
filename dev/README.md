# Dockerfile and setup needed for development.

The [dev dockerfile](./nvcr.Dockerfile) is based on NVCR pytorch container version nvcr.io/nvidia/pytorch:25.02-py3.

If the nvcr container is updated, ensure to check the torch and python installation version inside the dockerfile before pushing changes.