# app.py
import os, sys, threading, subprocess, tkinter as tk
from tkinter import filedialog, messagebox

# ====== AJUSTE AQUI ======
SCRIPT_PATH = r"C:\Automation_ICL_REURB\extract_cadaster.py"
# =========================

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Runner - extract_cadaster.py")
        self.geometry("760x520")

        self.pdf_path = tk.StringVar(value="")
        self.debug_regions = tk.BooleanVar(value=True)

        top = tk.Frame(self); top.pack(fill="x", padx=10, pady=10)
        tk.Button(top, text="Escolher PDF…", command=self.choose_pdf).pack(side="left")
        tk.Entry(top, textvariable=self.pdf_path, width=80).pack(side="left", padx=8)

        params = tk.Frame(self); params.pack(fill="x", padx=10)
        self.min_lotes = tk.StringVar(value="3")
        self.snap_tol  = tk.StringVar(value="6")
        self.min_area  = tk.StringVar(value="300")
        self.max_imov  = tk.StringVar(value="50")
        self.max_res   = tk.StringVar(value="50")

        for lbl, var in [("--min-lotes-in-poly", self.min_lotes),
                         ("--snap-tol", self.snap_tol),
                         ("--min-cycle-area", self.min_area),
                         ("--max-imovel", self.max_imov),
                         ("--max-res", self.max_res)]:
            f = tk.Frame(params); f.pack(side="left", padx=6)
            tk.Label(f, text=lbl).pack()
            tk.Entry(f, textvariable=var, width=8).pack()

        tk.Checkbutton(self, text="--debug-regions", variable=self.debug_regions).pack(anchor="w", padx=10, pady=4)

        actions = tk.Frame(self); actions.pack(fill="x", padx=10)
        self.btn_run = tk.Button(actions, text="Rodar", command=self.run_now)
        self.btn_run.pack(side="left")
        tk.Button(actions, text="Limpar log", command=lambda: self.log.delete("1.0","end")).pack(side="left", padx=6)

        box = tk.LabelFrame(self, text="Saída do script"); box.pack(fill="both", expand=True, padx=10, pady=10)
        self.log = tk.Text(box, height=18); self.log.pack(fill="both", expand=True)

    def choose_pdf(self):
        p = filedialog.askopenfilename(title="Selecione o PDF", filetypes=[("PDF","*.pdf")])
        if p: self.pdf_path.set(p)

    def run_now(self):
        script = SCRIPT_PATH
        pdf = self.pdf_path.get().strip()

        if not os.path.isfile(script):
            messagebox.showerror("Erro", f"Script não encontrado:\n{script}"); return
        if not pdf or not os.path.isfile(pdf):
            messagebox.showerror("Erro", f"PDF inválido:\n{pdf}"); return

        # arquivo de saída = mesmo nome do PDF, .xlsx, na mesma pasta
        base_name = os.path.splitext(os.path.basename(pdf))[0] + ".xlsx"
        out_dir = os.path.join(os.path.dirname(__file__), "results_testes")
        os.makedirs(out_dir, exist_ok=True)
        out_xlsx = os.path.join(out_dir, base_name)

        # monta os args (sem backticks; subprocess cuida de espaços)
        args = [
            sys.executable, script,                   # usa o mesmo Python/venv
            pdf,
            "--out", out_xlsx,
            "--min-lotes-in-poly", self.min_lotes.get(),
            "--snap-tol", self.snap_tol.get(),
            "--min-cycle-area", self.min_area.get(),
            "--max-imovel", self.max_imov.get(),
            "--max-res", self.max_res.get(),
        ]
        if self.debug_regions.get():
            args.append("--debug-regions")

        self.btn_run.config(state="disabled")
        self._append(f"> Executando:\n{' '.join(self._quote(a) for a in args)}\n\n")

        t = threading.Thread(target=self._run_proc, args=(args,), daemon=True)
        t.start()

    def _run_proc(self, args):
        try:
            env = os.environ.copy()
            env.update({
                "PYTHONUTF8": "1",           
                "PYTHONIOENCODING": "utf-8"  
            })
            proc = subprocess.Popen(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                env=env,                     # <<<<<< AQUI
            )
            for line in proc.stdout:
                self._append(line)
            code = proc.wait()
            self._append("\n✅ Finalizado.\n" if code == 0 else f"\n❌ Código de saída: {code}\n")
        except Exception as e:
            self._append(f"\n❌ Erro: {e}\n")
        finally:
            self.btn_run.config(state="normal")


    def _append(self, txt):
        self.after(0, lambda: (self.log.insert("end", txt), self.log.see("end")))

    @staticmethod
    def _quote(s): return f"\"{s}\"" if " " in s and not s.startswith("\"") else s

if __name__ == "__main__":
    App().mainloop()
