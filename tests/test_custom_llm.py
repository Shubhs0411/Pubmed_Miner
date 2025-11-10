# tests/test_custom_llm.py
# Integration test for the Custom LLM backend using a fake HTTP server.
# Generates CSV output like the real app.

import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import sys
from pathlib import Path
import importlib.util

# Ensure project root is on sys.path so that `from llm import ...` works
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Dynamically load the custom backend as part of the llm package
spec = importlib.util.spec_from_file_location("llm.custom", ROOT / "llm" / "custom.py")
custom_module = importlib.util.module_from_spec(spec)
sys.modules["llm.custom"] = custom_module
spec.loader.exec_module(custom_module)

# Import CSV export function
from pipeline.csv_export import flatten_to_rows

# More realistic fake response with multiple findings
FAKE_RESPONSE = {
    "completion": json.dumps([
        {
            "virus": "Chikungunya virus",
            "protein": "E1",
            "feature": {
                "name_or_label": "A226V",
                "type": "mutation_effect",
                "continuity": "point",
                "residue_positions": [],
                "specific_residues": [{"position": 226, "aa": "A→V"}],
                "variants": ["p.Ala226Val"],
                "motif_pattern": None,
            },
            "effect_or_function": {
                "description": "Enhances transmission by Aedes albopictus mosquitoes.",
                "category": "vector_adaptation",
                "direction": "increase",
                "evidence_level": "experimental",
            },
            "interactions": {
                "partner_protein": None,
                "interaction_type": None,
                "context": None,
            },
            "evidence_snippet": "The single-point mutation A226V in E1 increased transmission by Ae. albopictus.",
            "confidence": {"score_0_to_1": 0.95, "rationale": "Explicit residue position."},
        },
        {
            "virus": "Dengue virus",
            "protein": "NS3",
            "feature": {
                "name_or_label": "K128E",
                "type": "mutation_effect",
                "continuity": "point",
                "residue_positions": [],
                "specific_residues": [{"position": 128, "aa": "K→E"}],
                "variants": ["p.Lys128Glu"],
                "motif_pattern": None,
            },
            "effect_or_function": {
                "description": "Reduces viral replication efficiency.",
                "category": "replication",
                "direction": "decrease",
                "evidence_level": "experimental",
            },
            "interactions": {
                "partner_protein": None,
                "interaction_type": None,
                "context": None,
            },
            "evidence_snippet": "Mutation K128E in NS3 protein significantly reduced viral replication in cell culture.",
            "confidence": {"score_0_to_1": 0.88, "rationale": "Clear experimental evidence."},
        }
    ])
}


class FakeLLMHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        _ = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(FAKE_RESPONSE).encode("utf-8"))

    def log_message(self, format, *args):
        return


def main():
    print("🚀 Starting fake LLM server on localhost:5055...")
    server = HTTPServer(("localhost", 5055), FakeLLMHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print("✅ Fake server running\n")

    try:
        # Simulate a real paper text
        sample_text = """
        This paper discusses several mutations in viral proteins that affect transmission and replication.
        
        The single-point mutation A226V in E1 protein of Chikungunya virus increased transmission by 
        Aedes albopictus mosquitoes. This mutation has been extensively studied and shows clear 
        experimental evidence of enhanced vector adaptation.
        
        Additionally, mutation K128E in the NS3 protein of Dengue virus significantly reduced viral 
        replication in cell culture experiments. This finding suggests potential targets for antiviral 
        intervention strategies.
        """
        
        meta = {
            "api_url": "http://localhost:5055/v1/completions",
            "api_key": "",
            "model_name": "demo-model",
            "timeout": 30,
            "chunk_chars": 10000,
            "pmid": "12345678",
            "pmcid": "PMC123456",
        }

        print("📄 Running custom LLM extraction...")
        result = custom_module.run_on_paper(sample_text, meta=meta)
        
        print("\n📊 Raw LLM Result:")
        print(json.dumps(result, indent=2))
        
        # Simulate the batch format that the app uses
        batch_results = {
            "12345678": {
                "status": "ok",
                "pmcid": "PMC123456",
                "title": "Test Paper: Viral Mutations and Their Effects",
                "result": result,
            }
        }
        
        # Convert to CSV format
        print("\n🔄 Converting to CSV format...")
        df = flatten_to_rows(batch_results)
        
        # Save CSV
        output_csv = ROOT / "tests" / "test_custom_llm_output.csv"
        df.to_csv(output_csv, index=False, encoding="utf-8")
        
        print(f"\n✅ CSV saved to: {output_csv}")
        print(f"\n📋 CSV Preview ({len(df)} rows):")
        print("=" * 80)
        print(df.to_string(index=False))
        print("=" * 80)
        print(f"\n💾 Full CSV available at: {output_csv}")
        
    finally:
        print("\n🛑 Shutting down fake server...")
        server.shutdown()
        thread.join(timeout=1)
        print("✅ Done!")


if __name__ == "__main__":
    sys.exit(main())