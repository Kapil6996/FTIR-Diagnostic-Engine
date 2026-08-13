import re

with open("app.py", "r") as f:
    code = f.read()

target = """    def _update_progress(self, current, total, msg, metrics):
        # Update text console
        self.console.insert(tk.END, f"{msg}\n")
        self.console.see(tk.END)"""
replacement = """    def _update_progress(self, current, total, msg, metrics):
        # Update text console
        ext_source = metrics.get("extraction_source", "")
        if ext_source and ext_source != "None":
            msg = f"{msg} [Source: {ext_source}]"
        
        self.console.insert(tk.END, f"{msg}\n")
        self.console.see(tk.END)"""

code = code.replace(target, replacement)

with open("app.py", "w") as f:
    f.write(code)

print("app.py updated")
