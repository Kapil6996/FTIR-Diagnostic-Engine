"""
FTIR Results Browser & Manual Review Windows (src.review_viewer)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Provides two Toplevel Tkinter dialogs that open after the pipeline completes:

1. **FtirBrowserWindow** — Browse ALL processed FTIRs in a list.
   Select any FTIR to see the photos extracted from its attachment folder.

2. **ManualReviewWindow** — Browse only FTIRs flagged for manual review.
   Select an FTIR to see its photos, then manually assign an SBPR number
   using a dropdown and save button. Changes are written back to the
   output Excel in real time.
"""

import os
import glob
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from typing import List, Optional, Dict, Any

from PIL import Image, ImageTk  # Pillow for photo display

# Supported image extensions
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".gif", ".webp"}

from src.pipeline import FTIR_RECORDS_DIR


def _get_images_for_ftir(ftir_no: str, records_dir: str = FTIR_RECORDS_DIR) -> List[str]:
    """Return list of image file paths inside a given FTIR's attachment folder, including extracted PDF/Video frames."""
    from src.media_normalize import get_all_images_for_ftir
    folder = os.path.join(records_dir, str(ftir_no))
    return get_all_images_for_ftir(folder)


# ══════════════════════════════════════════════════════════════════════════════
#  1. FTIR BROWSER WINDOW — View All FTIRs & Their Photos
# ══════════════════════════════════════════════════════════════════════════════

class FtirBrowserWindow(tk.Toplevel):
    """
    A pop-up window showing a scrollable list of all processed FTIR records
    on the left. Clicking any FTIR shows its extracted photos on the right
    in a scrollable thumbnail gallery.
    """

    def __init__(self, parent, result_df, records_dir=FTIR_RECORDS_DIR,
                 output_path: str = "", sbpr_list: Optional[List[str]] = None):
        super().__init__(parent)
        self.title("📋 FTIR Results Browser — View All Processed Records")
        self.geometry("1100x700")
        self.minsize(900, 500)
        self.configure(bg="#EAEEF3")

        self.result_df = result_df
        self.records_dir = records_dir
        self.output_path = output_path
        self.sbpr_list = sbpr_list or []
        self._photo_refs = []  # prevent garbage collection of PhotoImage objects
        self._current_entry = None  # currently displayed FTIR entry

        self._build_ui()

    def _build_ui(self):
        # ── Header ───────────────────────────────────────────────────────
        header = tk.Frame(self, bg="#0F172A", padx=20, pady=14)
        header.pack(fill=tk.X)
        tk.Label(header, text="FTIR Results Browser", font=("Segoe UI", 16, "bold"),
                 bg="#0F172A", fg="#FFFFFF").pack(anchor=tk.W)
        tk.Label(header, text="Select any FTIR from the list to view its extracted photos and classification result.",
                 font=("Segoe UI", 10), bg="#0F172A", fg="#38BDF8").pack(anchor=tk.W, pady=(3, 0))

        # ── Main Paned Layout ────────────────────────────────────────────
        pane = tk.PanedWindow(self, orient=tk.HORIZONTAL, bg="#EAEEF3", sashwidth=6)
        pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # ── Left: FTIR List ──────────────────────────────────────────────
        left_frame = tk.Frame(pane, bg="#FFFFFF", bd=1, relief="solid")
        pane.add(left_frame, width=340)

        tk.Label(left_frame, text="FTIR Records", font=("Segoe UI", 12, "bold"),
                 bg="#FFFFFF", fg="#0F172A").pack(anchor=tk.W, padx=10, pady=(10, 5))

        # Search/filter bar
        search_frame = tk.Frame(left_frame, bg="#FFFFFF")
        search_frame.pack(fill=tk.X, padx=10, pady=(0, 5))
        tk.Label(search_frame, text="🔍", bg="#FFFFFF", font=("Segoe UI", 11)).pack(side=tk.LEFT)
        self.var_search = tk.StringVar()
        self.var_search.trace_add("write", lambda *_: self._filter_list())
        tk.Entry(search_frame, textvariable=self.var_search, font=("Segoe UI", 10),
                 relief="solid", bd=1).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))

        # Listbox with scrollbar
        list_container = tk.Frame(left_frame, bg="#FFFFFF")
        list_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=(0, 5))

        self.listbox = tk.Listbox(
            list_container, font=("Consolas", 11), selectmode=tk.SINGLE,
            bg="#F8FAFC", fg="#0F172A", selectbackground="#2563EB",
            selectforeground="#FFFFFF", activestyle="none", bd=0
        )
        sb = ttk.Scrollbar(list_container, orient=tk.VERTICAL, command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=sb.set)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        self.listbox.bind("<<ListboxSelect>>", self._on_select)

        # Populate list
        self._populate_list()

        # ── Right: Photo Gallery + Info ──────────────────────────────────
        right_frame = tk.Frame(pane, bg="#FFFFFF", bd=1, relief="solid")
        pane.add(right_frame)

        # Info panel at top
        self.info_frame = tk.Frame(right_frame, bg="#F1F5F9", padx=15, pady=10)
        self.info_frame.pack(fill=tk.X)

        self.lbl_ftir_title = tk.Label(self.info_frame, text="Select an FTIR from the list",
                                        font=("Segoe UI", 14, "bold"), bg="#F1F5F9", fg="#0F172A")
        self.lbl_ftir_title.pack(anchor=tk.W)

        self.lbl_ftir_info = tk.Label(self.info_frame, text="",
                                       font=("Segoe UI", 10), bg="#F1F5F9", fg="#334155",
                                       wraplength=700, justify=tk.LEFT)
        self.lbl_ftir_info.pack(anchor=tk.W, pady=(4, 0))

        # Photo gallery (scrollable canvas)
        gallery_container = tk.Frame(right_frame, bg="#FFFFFF")
        gallery_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.canvas = tk.Canvas(gallery_container, bg="#FFFFFF", highlightthickness=0)
        self.gallery_scrollbar = ttk.Scrollbar(gallery_container, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.gallery_scrollbar.set)

        self.gallery_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.gallery_inner = tk.Frame(self.canvas, bg="#FFFFFF")
        self.canvas_window = self.canvas.create_window((0, 0), window=self.gallery_inner, anchor="nw")

        self.gallery_inner.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfig(self.canvas_window, width=e.width))

        # No-photos placeholder
        self.lbl_no_photos = tk.Label(self.gallery_inner, text="No photos to display.\nSelect an FTIR from the left panel.",
                                       font=("Segoe UI", 12), bg="#FFFFFF", fg="#94A3B8")
        self.lbl_no_photos.pack(expand=True, pady=60)

    def _get_all_ftir_entries(self) -> List[Dict[str, Any]]:
        """Build list of FTIR entries from the result DataFrame."""
        entries = []
        # Find the FTIR column
        ftir_col = None
        for col in self.result_df.columns:
            cl = str(col).lower()
            if "ftir" in cl and ("no" in cl or "num" in cl or "id" in cl or cl == "ftir"):
                ftir_col = col
                break
        if ftir_col is None:
            for col in self.result_df.columns:
                if "ftir" in str(col).lower():
                    ftir_col = col
                    break

        for idx, row in self.result_df.iterrows():
            ftir_no = str(row.get(ftir_col, f"row_{idx}")).strip() if ftir_col else f"row_{idx}"
            defect = str(row.get("Defect_Type", "N/A"))
            sbpr = str(row.get("SBPR_Number", "N/A"))
            review = bool(row.get("Flag_For_Review", False))
            reason = str(row.get("Reason", ""))
            entries.append({
                "ftir_no": ftir_no,
                "defect": defect,
                "sbpr": sbpr,
                "review": review,
                "reason": reason,
                "row_idx": idx,
            })
        return entries

    def _populate_list(self):
        self._all_entries = self._get_all_ftir_entries()
        self.listbox.delete(0, tk.END)
        for e in self._all_entries:
            flag = " ⚠" if e["review"] else ""
            display = f"{e['ftir_no']}  →  {e['sbpr']}{flag}"
            self.listbox.insert(tk.END, display)

    def _filter_list(self):
        query = self.var_search.get().strip().lower()
        self.listbox.delete(0, tk.END)
        for e in self._all_entries:
            if query in e["ftir_no"].lower() or query in e["sbpr"].lower() or query in e["defect"].lower():
                flag = " ⚠" if e["review"] else ""
                display = f"{e['ftir_no']}  →  {e['sbpr']}{flag}"
                self.listbox.insert(tk.END, display)

    def _on_select(self, event):
        sel = self.listbox.curselection()
        if not sel:
            return
        # Find matching entry from filtered view
        display_text = self.listbox.get(sel[0])
        ftir_no = display_text.split("  →")[0].strip()
        entry = None
        for e in self._all_entries:
            if e["ftir_no"] == ftir_no:
                entry = e
                break
        if not entry:
            return
        self._show_ftir(entry)

    def _show_ftir(self, entry: dict):
        """Display info and photos for the selected FTIR."""
        ftir_no = entry["ftir_no"]
        self._current_entry = entry  # track for report dialog

        # Clear old report button if it exists
        if hasattr(self, '_btn_report') and self._btn_report.winfo_exists():
            self._btn_report.destroy()

        # Report button on the far right of the info_frame
        self._btn_report = tk.Button(self.info_frame, text="🚩 Report Wrong Classification",
                                     font=("Segoe UI", 10, "bold"), bg="#F59E0B", fg="#000000",
                                     activebackground="#D97706", cursor="hand2", relief="flat",
                                     padx=12, pady=3, command=self._open_report_dialog)
        self._btn_report.place(relx=1.0, rely=0.0, anchor="ne", x=-10, y=0)

        # Update info labels
        review_txt = "  ⚠️ FLAGGED FOR MANUAL REVIEW" if entry["review"] else ""
        self.lbl_ftir_title.config(text=f"FTIR: {ftir_no}{review_txt}")
        info_parts = [
            f"Defect Type: {entry['defect'].upper()}",
            f"SBPR Assigned: {entry['sbpr']}",
        ]
        if entry["reason"]:
            info_parts.append(f"Reason: {entry['reason']}")
        self.lbl_ftir_info.config(text="  |  ".join(info_parts))

        # Clear old photos
        for widget in self.gallery_inner.winfo_children():
            widget.destroy()
        self._photo_refs.clear()

        # Load images
        images = _get_images_for_ftir(ftir_no, self.records_dir)
        if not images:
            tk.Label(self.gallery_inner, text=f"No extracted photos found for FTIR {ftir_no}.\n\n"
                     f"(Folder: {os.path.join(self.records_dir, ftir_no)})",
                     font=("Segoe UI", 11), bg="#FFFFFF", fg="#94A3B8").pack(expand=True, pady=40)
            return

        tk.Label(self.gallery_inner, text=f"📸 {len(images)} photo(s) extracted:",
                 font=("Segoe UI", 11, "bold"), bg="#FFFFFF", fg="#0F172A").pack(anchor=tk.W, padx=10, pady=(10, 5))

        # Display images in a grid
        grid_frame = tk.Frame(self.gallery_inner, bg="#FFFFFF")
        grid_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        for i, img_path in enumerate(images):
            try:
                img = Image.open(img_path)
                # Thumbnail to reasonable size
                img.thumbnail((300, 300), Image.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                self._photo_refs.append(photo)

                card = tk.Frame(grid_frame, bg="#F1F5F9", bd=1, relief="solid", padx=8, pady=8)
                card.grid(row=i // 3, column=i % 3, padx=6, pady=6, sticky="nsew")

                tk.Label(card, image=photo, bg="#F1F5F9").pack()
                tk.Label(card, text=os.path.basename(img_path), font=("Segoe UI", 8),
                         bg="#F1F5F9", fg="#64748B", wraplength=280).pack(pady=(4, 0))
            except Exception:
                card = tk.Frame(grid_frame, bg="#FEF2F2", bd=1, relief="solid", padx=8, pady=8)
                card.grid(row=i // 3, column=i % 3, padx=6, pady=6, sticky="nsew")
                tk.Label(card, text=f"⚠ Could not load:\n{os.path.basename(img_path)}",
                         font=("Segoe UI", 9), bg="#FEF2F2", fg="#DC2626").pack(pady=20)

        for c in range(3):
            grid_frame.columnconfigure(c, weight=1)

    def _open_report_dialog(self):
        """Open the Report Misclassification dialog for the currently selected FTIR."""
        if not self._current_entry:
            messagebox.showwarning("No FTIR Selected", "Please select an FTIR first.")
            return
        ReportMisclassificationDialog(
            parent=self,
            entry=self._current_entry,
            result_df=self.result_df,
            output_path=self.output_path,
            sbpr_list=self.sbpr_list,
            records_dir=self.records_dir,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  2. MANUAL REVIEW WINDOW — Review Flagged FTIRs & Classify
# ══════════════════════════════════════════════════════════════════════════════

class ManualReviewWindow(tk.Toplevel):
    """
    A pop-up window showing only FTIRs flagged for manual review.
    Select an FTIR to see its photos. A dropdown lets the operator
    manually assign the correct SBPR number, which is saved back to
    the output Excel immediately.
    """

    def __init__(self, parent, result_df, output_path: str, sbpr_list: List[str],
                 records_dir=FTIR_RECORDS_DIR):
        super().__init__(parent)
        self.title("⚠️ Manual Review — Classify Flagged FTIR Records")
        self.geometry("1150x750")
        self.minsize(950, 550)
        self.configure(bg="#EAEEF3")

        self.result_df = result_df
        self.output_path = output_path
        self.sbpr_list = sbpr_list
        self.records_dir = records_dir
        self._photo_refs = []
        self._current_entry = None

        self._build_ui()

    def _build_ui(self):
        # ── Header ───────────────────────────────────────────────────────
        header = tk.Frame(self, bg="#7C2D12", padx=20, pady=14)
        header.pack(fill=tk.X)
        tk.Label(header, text="⚠️ Manual Review — Flagged FTIR Records", font=("Segoe UI", 16, "bold"),
                 bg="#7C2D12", fg="#FFFFFF").pack(anchor=tk.W)
        tk.Label(header, text="These FTIRs need human verification. View photos and assign the correct SBPR classification.",
                 font=("Segoe UI", 10), bg="#7C2D12", fg="#FED7AA").pack(anchor=tk.W, pady=(3, 0))

        # ── Main Paned Layout ────────────────────────────────────────────
        pane = tk.PanedWindow(self, orient=tk.HORIZONTAL, bg="#EAEEF3", sashwidth=6)
        pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # ── Left: Flagged FTIR List ──────────────────────────────────────
        left_frame = tk.Frame(pane, bg="#FFFFFF", bd=1, relief="solid")
        pane.add(left_frame, width=320)

        tk.Label(left_frame, text="Flagged for Review", font=("Segoe UI", 12, "bold"),
                 bg="#FFFFFF", fg="#B91C1C").pack(anchor=tk.W, padx=10, pady=(10, 5))

        # Counter
        review_entries = self._get_review_entries()
        self.lbl_count = tk.Label(left_frame,
                                   text=f"{len(review_entries)} record(s) need manual classification",
                                   font=("Segoe UI", 9, "italic"), bg="#FFFFFF", fg="#64748B")
        self.lbl_count.pack(anchor=tk.W, padx=10, pady=(0, 8))

        # Listbox
        list_container = tk.Frame(left_frame, bg="#FFFFFF")
        list_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=(0, 5))

        self.listbox = tk.Listbox(
            list_container, font=("Consolas", 11), selectmode=tk.SINGLE,
            bg="#FFF7ED", fg="#0F172A", selectbackground="#EA580C",
            selectforeground="#FFFFFF", activestyle="none", bd=0
        )
        sb = ttk.Scrollbar(list_container, orient=tk.VERTICAL, command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=sb.set)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        self.listbox.bind("<<ListboxSelect>>", self._on_select)
        self._populate_list()

        # ── Right: Photo Gallery + Classification Controls ───────────────
        right_frame = tk.Frame(pane, bg="#FFFFFF", bd=1, relief="solid")
        pane.add(right_frame)

        # Info panel
        self.info_frame = tk.Frame(right_frame, bg="#FFF7ED", padx=15, pady=10)
        self.info_frame.pack(fill=tk.X)

        self.lbl_ftir_title = tk.Label(self.info_frame, text="Select a flagged FTIR from the list",
                                        font=("Segoe UI", 14, "bold"), bg="#FFF7ED", fg="#0F172A")
        self.lbl_ftir_title.pack(anchor=tk.W)

        self.lbl_ftir_info = tk.Label(self.info_frame, text="",
                                       font=("Segoe UI", 10), bg="#FFF7ED", fg="#334155",
                                       wraplength=650, justify=tk.LEFT)
        self.lbl_ftir_info.pack(anchor=tk.W, pady=(4, 0))

        # ── Classification Controls ──────────────────────────────────────
        classify_frame = tk.Frame(right_frame, bg="#FFFBEB", padx=15, pady=12, bd=1, relief="solid")
        classify_frame.pack(fill=tk.X, padx=10, pady=(10, 5))

        tk.Label(classify_frame, text="Manual SBPR Classification:", font=("Segoe UI", 12, "bold"),
                 bg="#FFFBEB", fg="#92400E").pack(anchor=tk.W)

        controls_row = tk.Frame(classify_frame, bg="#FFFBEB")
        controls_row.pack(fill=tk.X, pady=(8, 0))

        tk.Label(controls_row, text="Assign SBPR:", font=("Segoe UI", 11),
                 bg="#FFFBEB", fg="#0F172A").pack(side=tk.LEFT)

        self.var_sbpr_choice = tk.StringVar()
        sbpr_options = ["— Select SBPR —"] + self.sbpr_list + ["uncertain", "non_rust"]
        self.combo_sbpr = ttk.Combobox(controls_row, textvariable=self.var_sbpr_choice,
                                        values=sbpr_options, state="readonly",
                                        font=("Consolas", 11), width=28)
        self.combo_sbpr.current(0)
        self.combo_sbpr.pack(side=tk.LEFT, padx=(10, 15))

        self.btn_save = tk.Button(
            controls_row, text="✅ Save Classification", font=("Segoe UI", 11, "bold"),
            bg="#16A34A", fg="#FFFFFF", activebackground="#15803D", activeforeground="#FFFFFF",
            padx=15, pady=4, relief="flat", cursor="hand2",
            command=self._save_classification
        )
        self.btn_save.pack(side=tk.LEFT, padx=(0, 10))

        self.lbl_save_status = tk.Label(controls_row, text="", font=("Segoe UI", 10, "bold"),
                                         bg="#FFFBEB", fg="#16A34A")
        self.lbl_save_status.pack(side=tk.LEFT)

        # ── Photo Gallery ────────────────────────────────────────────────
        gallery_container = tk.Frame(right_frame, bg="#FFFFFF")
        gallery_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.canvas = tk.Canvas(gallery_container, bg="#FFFFFF", highlightthickness=0)
        self.gallery_scrollbar = ttk.Scrollbar(gallery_container, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.gallery_scrollbar.set)

        self.gallery_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.gallery_inner = tk.Frame(self.canvas, bg="#FFFFFF")
        self.canvas_window = self.canvas.create_window((0, 0), window=self.gallery_inner, anchor="nw")

        self.gallery_inner.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfig(self.canvas_window, width=e.width))

        # Placeholder
        tk.Label(self.gallery_inner, text="Select a flagged FTIR to view photos and classify.",
                 font=("Segoe UI", 12), bg="#FFFFFF", fg="#94A3B8").pack(expand=True, pady=60)

    def _get_review_entries(self) -> List[Dict[str, Any]]:
        """Build list of FTIR entries flagged for review."""
        entries = []
        ftir_col = None
        for col in self.result_df.columns:
            cl = str(col).lower()
            if "ftir" in cl and ("no" in cl or "num" in cl or "id" in cl or cl == "ftir"):
                ftir_col = col
                break
        if ftir_col is None:
            for col in self.result_df.columns:
                if "ftir" in str(col).lower():
                    ftir_col = col
                    break

        for idx, row in self.result_df.iterrows():
            if not bool(row.get("Flag_For_Review", False)):
                continue
            ftir_no = str(row.get(ftir_col, f"row_{idx}")).strip() if ftir_col else f"row_{idx}"
            defect = str(row.get("Defect_Type", "N/A"))
            sbpr = str(row.get("SBPR_Number", "N/A"))
            reason = str(row.get("Reason", ""))
            entries.append({
                "ftir_no": ftir_no,
                "defect": defect,
                "sbpr": sbpr,
                "reason": reason,
                "row_idx": idx,
            })
        return entries

    def _populate_list(self):
        self._review_entries = self._get_review_entries()
        self.listbox.delete(0, tk.END)
        for e in self._review_entries:
            display = f"⚠ {e['ftir_no']}  →  {e['sbpr']}"
            self.listbox.insert(tk.END, display)

    def _on_select(self, event):
        sel = self.listbox.curselection()
        if not sel:
            return
        entry = self._review_entries[sel[0]]
        self._current_entry = entry
        self._show_ftir(entry)

    def _show_ftir(self, entry: dict):
        """Display info, photos, and classification controls for a flagged FTIR."""
        ftir_no = entry["ftir_no"]

        self.lbl_ftir_title.config(text=f"⚠️ FTIR: {ftir_no} — Needs Manual Classification")
        info_parts = [
            f"Defect: {entry['defect'].upper()}",
            f"AI Suggestion: {entry['sbpr']}",
        ]
        if entry["reason"]:
            info_parts.append(f"Reason: {entry['reason']}")
        self.lbl_ftir_info.config(text="  |  ".join(info_parts))

        # Pre-select AI suggestion in dropdown
        if entry["sbpr"] in self.sbpr_list:
            self.var_sbpr_choice.set(entry["sbpr"])
        else:
            self.combo_sbpr.current(0)
        self.lbl_save_status.config(text="")

        # Clear and reload photos
        for widget in self.gallery_inner.winfo_children():
            widget.destroy()
        self._photo_refs.clear()

        images = _get_images_for_ftir(ftir_no, self.records_dir)
        if not images:
            tk.Label(self.gallery_inner, text=f"No extracted photos found for FTIR {ftir_no}.\n\n"
                     f"(Folder: {os.path.join(self.records_dir, ftir_no)})",
                     font=("Segoe UI", 11), bg="#FFFFFF", fg="#94A3B8").pack(expand=True, pady=40)
            return

        tk.Label(self.gallery_inner, text=f"📸 {len(images)} photo(s) — Review these to decide the correct SBPR:",
                 font=("Segoe UI", 11, "bold"), bg="#FFFFFF", fg="#0F172A").pack(anchor=tk.W, padx=10, pady=(10, 5))

        grid_frame = tk.Frame(self.gallery_inner, bg="#FFFFFF")
        grid_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        for i, img_path in enumerate(images):
            try:
                img = Image.open(img_path)
                img.thumbnail((300, 300), Image.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                self._photo_refs.append(photo)

                card = tk.Frame(grid_frame, bg="#FFF7ED", bd=1, relief="solid", padx=8, pady=8)
                card.grid(row=i // 3, column=i % 3, padx=6, pady=6, sticky="nsew")

                tk.Label(card, image=photo, bg="#FFF7ED").pack()
                tk.Label(card, text=os.path.basename(img_path), font=("Segoe UI", 8),
                         bg="#FFF7ED", fg="#64748B", wraplength=280).pack(pady=(4, 0))
            except Exception:
                card = tk.Frame(grid_frame, bg="#FEF2F2", bd=1, relief="solid", padx=8, pady=8)
                card.grid(row=i // 3, column=i % 3, padx=6, pady=6, sticky="nsew")
                tk.Label(card, text=f"⚠ Could not load:\n{os.path.basename(img_path)}",
                         font=("Segoe UI", 9), bg="#FEF2F2", fg="#DC2626").pack(pady=20)

        for c in range(3):
            grid_frame.columnconfigure(c, weight=1)

    def _save_classification(self):
        """Save the manually assigned SBPR back to the DataFrame and output Excel."""
        if self._current_entry is None:
            messagebox.showwarning("No FTIR Selected", "Please select a flagged FTIR first.")
            return

        choice = self.var_sbpr_choice.get()
        if choice == "— Select SBPR —":
            messagebox.showwarning("No SBPR Selected", "Please select an SBPR number from the dropdown before saving.")
            return

        row_idx = self._current_entry["row_idx"]
        old_sbpr = self._current_entry["sbpr"]

        # Update DataFrame
        self.result_df.at[row_idx, "SBPR_Number"] = choice
        self.result_df.at[row_idx, "Flag_For_Review"] = False
        self.result_df.at[row_idx, "Reason"] = (
            f"Manually classified by operator (was: {old_sbpr}). "
            f"Original AI reason: {self.result_df.at[row_idx, 'Reason']}"
        )

        # Save to Excel
        try:
            from src.excel_io import write_output_sheet
            write_output_sheet(self.result_df, self.output_path)
            self.lbl_save_status.config(text=f"✔ Saved! {self._current_entry['ftir_no']} → {choice}", fg="#16A34A")

            # Update the entry and refresh list
            self._current_entry["sbpr"] = choice
            self._populate_list()

            # Update count
            remaining = len(self._review_entries)
            self.lbl_count.config(text=f"{remaining} record(s) still need manual classification")

        except Exception as e:
            self.lbl_save_status.config(text=f"✗ Save failed!", fg="#DC2626")
            messagebox.showerror("Save Error", f"Could not write to output Excel:\n{e}")


# ══════════════════════════════════════════════════════════════════════════════
#  3. REPORT MISCLASSIFICATION DIALOG
# ══════════════════════════════════════════════════════════════════════════════

class ReportMisclassificationDialog(tk.Toplevel):
    """
    A dialog window that allows the user to report a wrong classification
    for a specific FTIR record. Supports reporting both Model 1 (rust/non-rust)
    and Model 2 (SBPR code) errors. Saves corrections to a persistent JSON
    database and copies images to learning folders for future retraining.
    """

    def __init__(self, parent, entry: dict, result_df, output_path: str,
                 sbpr_list: List[str], records_dir: str = FTIR_RECORDS_DIR):
        super().__init__(parent)
        self.title(f"🚩 Report Wrong Classification — {entry['ftir_no']}")
        self.geometry("600x580")
        self.minsize(550, 520)
        self.configure(bg="#FFFBEB")
        self.resizable(True, True)

        self.entry = entry
        self.result_df = result_df
        self.output_path = output_path
        self.sbpr_list = sbpr_list
        self.records_dir = records_dir

        self._build_ui()

        # Make modal
        self.transient(parent)
        self.grab_set()

    def _build_ui(self):
        ftir_no = self.entry["ftir_no"]
        defect = self.entry["defect"].lower()
        sbpr = self.entry["sbpr"]

        # ── Header ────────────────────────────────────────────────────────
        header = tk.Frame(self, bg="#92400E", padx=16, pady=12)
        header.pack(fill=tk.X)
        tk.Label(header, text=f"Report Wrong Classification",
                 font=("Segoe UI", 14, "bold"), bg="#92400E", fg="#FFFFFF").pack(anchor=tk.W)
        tk.Label(header, text=f"FTIR: {ftir_no}  |  Current: {defect.upper()} → {sbpr}",
                 font=("Segoe UI", 10), bg="#92400E", fg="#FDE68A").pack(anchor=tk.W, pady=(3, 0))

        # ── Scrollable Content ────────────────────────────────────────────
        content = tk.Frame(self, bg="#FFFBEB", padx=20, pady=15)
        content.pack(fill=tk.BOTH, expand=True)

        # ── Section 1: Error Type ─────────────────────────────────────────
        tk.Label(content, text="What went wrong?",
                 font=("Segoe UI", 12, "bold"), bg="#FFFBEB", fg="#1E293B").pack(anchor=tk.W, pady=(0, 8))

        self.var_error_type = tk.StringVar(value="")

        # Model 1 options
        m1_frame = tk.LabelFrame(content, text="  Model 1 — Rust Detection Error  ",
                                  font=("Segoe UI", 10, "bold"), bg="#FEF3C7",
                                  fg="#92400E", padx=12, pady=8)
        m1_frame.pack(fill=tk.X, pady=(0, 8))

        tk.Radiobutton(m1_frame, text="Misclassified as RUST (it's actually Non-Rust)",
                       variable=self.var_error_type, value="model1_wrongly_rust",
                       font=("Segoe UI", 10), bg="#FEF3C7", fg="#1E293B",
                       activebackground="#FEF3C7", anchor=tk.W,
                       command=self._on_error_type_change).pack(fill=tk.X)

        tk.Radiobutton(m1_frame, text="Misclassified as NON-RUST (it's actually Rust)",
                       variable=self.var_error_type, value="model1_wrongly_nonrust",
                       font=("Segoe UI", 10), bg="#FEF3C7", fg="#1E293B",
                       activebackground="#FEF3C7", anchor=tk.W,
                       command=self._on_error_type_change).pack(fill=tk.X)

        # Model 2 options
        m2_frame = tk.LabelFrame(content, text="  Model 2 — SBPR Classification Error  ",
                                  font=("Segoe UI", 10, "bold"), bg="#FEF3C7",
                                  fg="#92400E", padx=12, pady=8)
        m2_frame.pack(fill=tk.X, pady=(0, 12))

        tk.Radiobutton(m2_frame, text="Rust detection was correct, but wrong SBPR assigned",
                       variable=self.var_error_type, value="model2_wrong_sbpr",
                       font=("Segoe UI", 10), bg="#FEF3C7", fg="#1E293B",
                       activebackground="#FEF3C7", anchor=tk.W,
                       command=self._on_error_type_change).pack(fill=tk.X)

        # Correct SBPR dropdown (hidden by default, shown when model2 is selected)
        self.sbpr_select_frame = tk.Frame(m2_frame, bg="#FEF3C7")
        tk.Label(self.sbpr_select_frame, text="Correct SBPR:",
                 font=("Segoe UI", 10), bg="#FEF3C7", fg="#1E293B").pack(side=tk.LEFT, padx=(20, 5))
        self.combo_correct_sbpr = ttk.Combobox(
            self.sbpr_select_frame, state="readonly", font=("Segoe UI", 10),
            values=["— Select correct SBPR —"] + self.sbpr_list, width=28
        )
        self.combo_correct_sbpr.current(0)
        self.combo_correct_sbpr.pack(side=tk.LEFT, padx=5)
        # Initially hidden
        self.sbpr_select_frame.pack_forget()

        # ── Section 2: Reason ─────────────────────────────────────────────
        tk.Label(content, text="Why is this a wrong classification?",
                 font=("Segoe UI", 12, "bold"), bg="#FFFBEB", fg="#1E293B").pack(anchor=tk.W, pady=(5, 5))

        self.txt_reason = scrolledtext.ScrolledText(
            content, font=("Segoe UI", 10), height=4, width=50,
            wrap=tk.WORD, relief="solid", bd=1
        )
        self.txt_reason.pack(fill=tk.X, pady=(0, 12))

        # ── Section 3: Submit ─────────────────────────────────────────────
        btn_frame = tk.Frame(content, bg="#FFFBEB")
        btn_frame.pack(fill=tk.X)

        self.btn_submit = tk.Button(
            btn_frame, text="✅ Submit Report & Save for Learning",
            font=("Segoe UI", 12, "bold"), bg="#16A34A", fg="#FFFFFF",
            activebackground="#15803D", cursor="hand2", relief="flat",
            padx=20, pady=8, command=self._submit_report
        )
        self.btn_submit.pack(side=tk.LEFT)

        tk.Button(
            btn_frame, text="Cancel",
            font=("Segoe UI", 10), bg="#E5E7EB", fg="#374151",
            cursor="hand2", relief="flat", padx=16, pady=6,
            command=self.destroy
        ).pack(side=tk.RIGHT)

        # Status label
        self.lbl_status = tk.Label(content, text="", font=("Segoe UI", 10), bg="#FFFBEB")
        self.lbl_status.pack(anchor=tk.W, pady=(8, 0))

    def _on_error_type_change(self):
        """Show/hide the SBPR dropdown based on error type selection."""
        if self.var_error_type.get() == "model2_wrong_sbpr":
            self.sbpr_select_frame.pack(fill=tk.X, pady=(5, 0))
        else:
            self.sbpr_select_frame.pack_forget()

    def _submit_report(self):
        """Validate inputs, save the correction, and update the output Excel."""
        error_type = self.var_error_type.get()
        if not error_type:
            messagebox.showwarning("Missing Selection",
                                   "Please select what went wrong (Model 1 or Model 2 error).",
                                   parent=self)
            return

        reason_text = self.txt_reason.get("1.0", tk.END).strip()
        if not reason_text:
            messagebox.showwarning("Missing Reason",
                                   "Please explain why this is a wrong classification.",
                                   parent=self)
            return

        # Determine the correct label based on error type
        if error_type == "model1_wrongly_rust":
            correct_label = "non_rust"
            original_pred = self.entry["defect"]
        elif error_type == "model1_wrongly_nonrust":
            correct_label = "rust"
            original_pred = self.entry["defect"]
        elif error_type == "model2_wrong_sbpr":
            correct_sbpr = self.combo_correct_sbpr.get()
            if correct_sbpr == "— Select correct SBPR —":
                messagebox.showwarning("Missing SBPR",
                                       "Please select the correct SBPR code from the dropdown.",
                                       parent=self)
                return
            correct_label = correct_sbpr
            original_pred = self.entry["sbpr"]
        else:
            return

        ftir_no = self.entry["ftir_no"]

        # Gather image paths and metadata for learning
        image_paths = _get_images_for_ftir(ftir_no, self.records_dir)
        row_idx = self.entry.get("row_idx")
        metadata = {}
        if row_idx is not None and row_idx in self.result_df.index:
            row_data = self.result_df.loc[row_idx]
            for col in ["Subject (English)", "Customer Complaint", "Mileage - Using Time",
                        "Masked Model", "Masked VIN", "Defect_Type", "SBPR_Number", "Reason"]:
                if col in row_data.index:
                    metadata[col] = str(row_data[col])

        # Save correction to persistent JSON database
        try:
            from src.corrections import save_correction
            save_correction(
                ftir_no=ftir_no,
                correction_type=error_type,
                original_prediction=original_pred,
                correct_label=correct_label,
                user_reason=reason_text,
                image_paths=image_paths,
                metadata=metadata,
            )
        except Exception as e:
            messagebox.showerror("Save Error",
                                 f"Could not save correction to database:\n{e}",
                                 parent=self)
            return

        # Update the result DataFrame and re-save the Excel file
        try:
            if row_idx is not None and row_idx in self.result_df.index:
                old_defect = self.result_df.at[row_idx, "Defect_Type"]
                old_sbpr = self.result_df.at[row_idx, "SBPR_Number"]
                old_reason = str(self.result_df.at[row_idx, "Reason"])

                if error_type in ("model1_wrongly_rust", "model1_wrongly_nonrust"):
                    self.result_df.at[row_idx, "Defect_Type"] = correct_label
                    if correct_label == "non_rust":
                        self.result_df.at[row_idx, "SBPR_Number"] = "N/A"
                    self.result_df.at[row_idx, "Flag_For_Review"] = False
                    self.result_df.at[row_idx, "Reason"] = (
                        f"[CORRECTED by operator] Was: {old_defect}/{old_sbpr}. "
                        f"Correction: {correct_label}. User reason: {reason_text}. "
                        f"Original AI reason: {old_reason}"
                    )
                elif error_type == "model2_wrong_sbpr":
                    self.result_df.at[row_idx, "SBPR_Number"] = correct_label
                    self.result_df.at[row_idx, "Flag_For_Review"] = False
                    self.result_df.at[row_idx, "Reason"] = (
                        f"[CORRECTED by operator] SBPR was: {old_sbpr}, now: {correct_label}. "
                        f"User reason: {reason_text}. "
                        f"Original AI reason: {old_reason}"
                    )

                # Re-save Excel
                if self.output_path and os.path.isdir(os.path.dirname(self.output_path)):
                    from src.excel_io import write_output_sheet
                    write_output_sheet(self.result_df, self.output_path)

        except Exception as e:
            # Non-fatal: correction is already saved to JSON even if Excel write fails
            messagebox.showwarning("Excel Warning",
                                    f"Correction saved to learning database, but could not "
                                    f"update Excel file:\n{e}",
                                    parent=self)

        # Show success
        self.lbl_status.config(
            text=f"✅ Report saved! Images copied to learning folder for FTIR {ftir_no}.",
            fg="#16A34A"
        )
        self.btn_submit.config(state=tk.DISABLED, text="✅ Report Submitted", bg="#86EFAC")

        # Auto-close after 2 seconds
        self.after(2000, self.destroy)
