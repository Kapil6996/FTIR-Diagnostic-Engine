import re

with open("app.py", "r") as f:
    code = f.read()

# 1. Add scrollable container in __init__
init_target = """        # Apply modern Tk styling
        self._setup_styles()

        # Build interactive interface sections"""

init_replacement = """        # Apply modern Tk styling
        self._setup_styles()
        
        # ── Build scrollable master container ──────────────────────────────
        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0, bg="#EAEEF3")
        self.v_scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.main_container = ttk.Frame(self.canvas, style="TFrame")
        
        self.main_container.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas_window = self.canvas.create_window((0, 0), window=self.main_container, anchor="nw")
        self.canvas.configure(yscrollcommand=self.v_scrollbar.set)
        
        # Expand canvas window width on resize
        def _configure_canvas(event):
            self.canvas.itemconfig(self.canvas_window, width=event.width)
        self.canvas.bind("<Configure>", _configure_canvas)
        
        # Mousewheel support
        def _on_mousewheel(event):
            if sys.platform == 'darwin':
                self.canvas.yview_scroll(int(-1*(event.delta)), "units")
            else:
                self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        self.canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        self.canvas.pack(side="left", fill="both", expand=True)
        self.v_scrollbar.pack(side="right", fill="y")

        # Build interactive interface sections (now inside self.main_container)"""

code = code.replace(init_target, init_replacement)

# 2. Update widget parents from self to self.main_container
code = code.replace('header = ttk.Frame(self, style="Header.TFrame"', 'header = ttk.Frame(self.main_container, style="Header.TFrame"')
code = code.replace('frame = ttk.Frame(self, style="Card.TFrame"', 'frame = ttk.Frame(self.main_container, style="Card.TFrame"')
code = code.replace('dash = ttk.Frame(self, padding=(15, 0))', 'dash = ttk.Frame(self.main_container, padding=(15, 0))')
code = code.replace('action_frame = ttk.Frame(self, padding=(15, 8))', 'action_frame = ttk.Frame(self.main_container, padding=(15, 8))')
code = code.replace('console_box = ttk.Frame(self, style="Card.TFrame"', 'console_box = ttk.Frame(self.main_container, style="Card.TFrame"')
code = code.replace('footer = ttk.Frame(self, padding=(15, 5))', 'footer = ttk.Frame(self.main_container, padding=(15, 5))')

with open("app.py", "w") as f:
    f.write(code)

print("Patch applied to app.py")
