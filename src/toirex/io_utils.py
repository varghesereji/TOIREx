import tkinter as tk
from tkinter import messagebox
from astroquery.simbad import Simbad
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np


def launch_simbad_gui():
    """Launch a SIMBAD GUI and return user-entered or queried results as a dictionary."""

    # Use the updated field names
    Simbad.add_votable_fields("ra", "dec", "pmra", "pmdec")

    # Hidden root for safety
    root_created = False
    root = tk._default_root
    if root is None:
        root = tk.Tk()
        root.withdraw()
        root_created = True

    # Dictionary to store results
    results = {"name": "", "ra": None, "dec": None, "pmra": None, "pmdec": None}

    def clear_fields():
        for e in (entry_ra, entry_dec, entry_pmra, entry_pmdec):
            e.delete(0, tk.END)

    def safe_to_float(value):
        if value is None or np.ma.is_masked(value):
            return None
        try:
            return float(value)
        except Exception:
            try:
                return float(str(value).strip())
            except Exception:
                return None

    def check_simbad(event=None):
        name = entry_name.get().strip()
        if not name:
            messagebox.showwarning("Input Error", "Please enter a target name.")
            return

        btn_check.config(state="disabled")
        gui.update_idletasks()

        try:
            result = Simbad.query_object(name)
            if result is None or len(result) == 0:
                clear_fields()
                messagebox.showinfo("Not found", f"'{name}' not found in SIMBAD.")
                return

            row = result[0]
            ra = dec = pmra = pmdec = None
            for key in row.colnames:
                k = key.lower()
                if k == "ra":
                    ra = safe_to_float(row[key])
                elif k == "dec":
                    dec = safe_to_float(row[key])
                elif k == "pmra":
                    pmra = safe_to_float(row[key])
                elif k == "pmdec":
                    pmdec = safe_to_float(row[key])

            if ra is not None and dec is not None:
                coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame="icrs")
                ra_str = coord.ra.to_string(unit=u.hour, sep=":", precision=2)
                dec_str = coord.dec.to_string(unit=u.deg, sep=":", precision=2,
                                              alwayssign=True)
            else:
                ra_str = dec_str = ""

            def fill(entry, value):
                entry.delete(0, tk.END)
                entry.insert(0, value)

            fill(entry_ra, str(ra_str))
            fill(entry_dec, str(dec_str))
            fill(entry_pmra, str(pmra) if pmra is not None else "")
            fill(entry_pmdec, str(pmdec) if pmdec is not None else "")

            # messagebox.showinfo("Success", f"'{name}' found in SIMBAD. Fields filled.")

        except Exception as e:
            messagebox.showerror("Error", f"Query failed:\n{e}")
        finally:
            btn_check.config(state="normal")

    def on_done():
        # Gather the final values
        results["name"] = entry_name.get().strip()
        results["ra"] = entry_ra.get()
        results["dec"] = entry_dec.get()
        results["pmra"] = safe_to_float(entry_pmra.get())
        results["pmdec"] = safe_to_float(entry_pmdec.get())
        gui.destroy()

    # ---------------- GUI Layout ----------------
    gui = tk.Toplevel(root)
    gui.title("SIMBAD Query Tool")
    gui.geometry("420x260")
    gui.resizable(False, False)

    padx, pady = 8, 6

    tk.Label(gui, text="Target Name:", font=("Arial", 11)).grid(row=0, column=0, sticky="e", padx=padx, pady=pady)
    entry_name = tk.Entry(gui, width=30, font=("Arial", 11))
    entry_name.grid(row=0, column=1, padx=padx, pady=pady)
    entry_name.focus_set()

    tk.Label(gui, text="RA (deg):", font=("Arial", 11)).grid(row=1, column=0, sticky="e", padx=padx, pady=pady)
    entry_ra = tk.Entry(gui, width=28, font=("Arial", 11))
    entry_ra.grid(row=1, column=1, padx=padx, pady=pady)

    tk.Label(gui, text="Dec (deg):", font=("Arial", 11)).grid(row=2, column=0, sticky="e", padx=padx, pady=pady)
    entry_dec = tk.Entry(gui, width=28, font=("Arial", 11))
    entry_dec.grid(row=2, column=1, padx=padx, pady=pady)

    tk.Label(gui, text="PMRA (mas/yr):", font=("Arial", 11)).grid(row=3, column=0, sticky="e", padx=padx, pady=pady)
    entry_pmra = tk.Entry(gui, width=28, font=("Arial", 11))
    entry_pmra.grid(row=3, column=1, padx=padx, pady=pady)

    tk.Label(gui, text="PMDEC (mas/yr):", font=("Arial", 11)).grid(row=4, column=0, sticky="e", padx=padx, pady=pady)
    entry_pmdec = tk.Entry(gui, width=28, font=("Arial", 11))
    entry_pmdec.grid(row=4, column=1, padx=padx, pady=pady)

    btn_frame = tk.Frame(gui)
    btn_frame.grid(row=5, column=0, columnspan=2, pady=(12, 8))

    btn_check = tk.Button(btn_frame, text="Check SIMBAD", width=14, font=("Arial", 11, "bold"), command=check_simbad)
    btn_check.pack(side="left", padx=6)

    btn_clear = tk.Button(btn_frame, text="Clear Fields", width=12, font=("Arial", 11), command=clear_fields)
    btn_clear.pack(side="left", padx=6)

    btn_done = tk.Button(btn_frame, text="Done", width=10, font=("Arial", 11), command=on_done)
    btn_done.pack(side="left", padx=6)

    entry_name.bind("<Return>", check_simbad)

    gui.protocol("WM_DELETE_WINDOW", on_done)
    gui.wait_window()

    if root_created:
        root.destroy()

    return results

# result = launch_simbad_gui()
# print(result)
