# Quick Start

Q/A microservice powered by a GPT-2 model, expertly fine-tuned for Persian-language contexts. This solution delivers accurate, context-aware responses, tailored specifically to the nuances of Persian dialogue and communication.

```sh
git clone git@github.com:vhidvz/question-answering.git
cd question-answering && docker-compose up -d
```

**Docker Hub:**

```sh
docker run -p 8000:8000 vhidvz/question-answering:latest
```

Endpoints are fully documented using OpenAPI Specification 3 (OAS3) at:

- ReDoc: <http://localhost:8000/redoc>
- Swagger: <http://localhost:8000/docs>

> Note: To enable in-memory document storage, simply remove the `ELASTICSEARCH_*` environment variables.

## Documentation

To generate the documentation for the python model, execute the following command:

```sh
pdoc --output-dir docs model.py
```
