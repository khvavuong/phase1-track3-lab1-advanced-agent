from __future__ import annotations
import json
import os
from pathlib import Path
import typer
from rich import print
from dotenv import load_dotenv
from src.reflexion_lab.agents import ReActAgent, ReflexionAgent
from src.reflexion_lab.provider import OpenAIProvider
from src.reflexion_lab.reporting import build_report, save_report
from src.reflexion_lab.utils import load_dataset, save_jsonl
app = typer.Typer(add_completion=False)

@app.command()
def main(
    dataset: str = "data/hotpot_mini.json",
    out_dir: str = "outputs/sample_run",
    reflexion_attempts: int = 3,
    real: bool = False,
    react_model: str = "",
    reflexion_model: str = "",
) -> None:
    load_dotenv()
    if not real:
        raise typer.BadParameter("Run with --real true to execute real OpenAI calls.")

    examples = load_dataset(dataset)
    selected_react_model = (os.getenv("OPENAI_MODEL_REACT"))
    selected_reflexion_model = (os.getenv("OPENAI_MODEL_REFLEXION"))

    react_provider = OpenAIProvider(model=selected_react_model)
    reflexion_provider = OpenAIProvider(model=selected_reflexion_model)
    react = ReActAgent(provider=react_provider)
    reflexion = ReflexionAgent(max_attempts=reflexion_attempts, provider=reflexion_provider)
    react_records = [react.run(example) for example in examples]
    reflexion_records = [reflexion.run(example) for example in examples]
    all_records = react_records + reflexion_records
    out_path = Path(out_dir)
    save_jsonl(out_path / "react_runs.jsonl", react_records)
    save_jsonl(out_path / "reflexion_runs.jsonl", reflexion_records)
    report = build_report(
        all_records,
        dataset_name=Path(dataset).name,
        mode=f"openai_api:react={selected_react_model};reflexion={selected_reflexion_model}",
    )
    json_path, md_path = save_report(report, out_path)
    print(f"[green]Saved[/green] {json_path}")
    print(f"[green]Saved[/green] {md_path}")
    print(json.dumps(report.summary, indent=2))

if __name__ == "__main__":
    app()
