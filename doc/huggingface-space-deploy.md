# Deploy Duka AI to a Hugging Face Space

This follows the standard **Space → Git → Push → Secrets** flow, mapped to **this repository** (`app.py`, `config.py`, AMD / OpenAI-compatible LLM).

## 1. Create the Space

1. On [Hugging Face](https://huggingface.co), **New** → **Space**.
2. **Owner:** use your account or the **hackathon org** (e.g. `lablab-ai-amd-developer-hackathon`) if submissions must live there.
3. **Space name:** e.g. `duka_ai`.
4. **License:** e.g. **MIT** (this repo includes a root `LICENSE`).
5. Keep the Space **Public** if you want shareable demo links / hackathon visibility.
6. **SDK:** choose **Docker** if the UI does not offer Streamlit alone — this repo ships a **`Dockerfile`** that runs Streamlit on port **7860**.

Click **Create Space**.

## 2. Git (already set up in this project)

From your project folder:

```bash
git status
```

This repo is already a Git repo with `main`. If you clone fresh:

```bash
git clone https://github.com/Emmzy550/Duka-ai.git
cd Duka-ai
```

## 3. Add the Hugging Face remote

Replace with **your** Space URL:

```bash
git remote add hf https://huggingface.co/spaces/<org-or-user>/<space-name>
```

Example (hackathon org):

```text
https://huggingface.co/spaces/lablab-ai-amd-developer-hackathon/duka_ai
```

If `hf` already exists:

```bash
git remote set-url hf https://huggingface.co/spaces/<org-or-user>/<space-name>
```

## 4. Stage, commit, push

Ensure **`requirements.txt`**, **`app.py`**, **`Dockerfile`**, and **`README.md`** are committed.

```bash
git add -A
git commit -m "Deploy: Space-ready files"
git push origin main          # GitHub
git push hf main              # first time may need --force-with-lease if the Space was created from a template
```

Authenticate with Hugging Face (**write** token). If histories diverged (template vs your code):

```bash
hf auth login --add-to-git-credential   # once
git push hf main --force-with-lease       # only when you intend to replace the Space repo with this project
```

You must have **write** access to that org Space or pushes will be rejected.

## 5. Configure secrets on Hugging Face

In the Space → **Settings** → **Variables and secrets**, add **Repository secrets** matching `config.py` / `.env.example`.

| Tutorial wording | Duka AI variable | Purpose |
|------------------|------------------|---------|
| VLM / server base URL | `AMD_BASE_URL` | OpenAI-compatible API base URL (e.g. AMD MI300X vLLM `http://host:port/v1`) |
| Model name | `AMD_MODEL` | Model id served by that endpoint (e.g. `Qwen/Qwen2.5-7B-Instruct`) |
| API key (if required) | `AMD_API_KEY` | Bearer/token your gateway expects (use a secret, not a public default) |

Also set:

| Variable | Notes |
|----------|--------|
| `LLM_PROVIDER` | `amd` (default) to use `AMD_*`; use `openai` only if you point to another OpenAI-compatible stack (`OPENAI_API_KEY`, `OPENAI_BASE_URL`, `MODEL_NAME`). |
| `TAVILY_API_KEY` | Optional — enables live market web search in Market Intel. |
| `APP_ENV` | e.g. `production` |

Never commit real keys to GitHub; keep them **only** in HF secrets (and local `.env` for development).

## 6. Smoke-test

Open the Space URL, run **Try Demo Analysis** or upload a sample file, and confirm LLM calls succeed (sidebar / logs if errors).
