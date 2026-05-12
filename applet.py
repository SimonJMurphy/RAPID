import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Button, CheckButtons, RadioButtons
from matplotlib import rcParams
import scipy as sp
from scipy.optimize import fsolve
from matplotlib import patheffects as pe
import corner


rcParams["xtick.direction"] = "in"
rcParams["ytick.direction"] = "in"
rcParams["xtick.top"] = True
rcParams["ytick.right"] = True
rcParams["font.size"] = 14


class InteractiveHRD:
    def __init__(self, df, solar_tracks, default_state, default_xlim, default_ylim, 
        weight_dist=3, bkg_thin_amount=50, teff_col="bin_teff", lum_col="bin_lum",
        ):
        # -----------------------------
        # Store state and defaults
        # -----------------------------
        self.data = df
        self.state = default_state.copy()
        self.default_state = default_state
        self.default_xlim = default_xlim
        self.default_ylim = default_ylim
        self.solar_tracks = solar_tracks
        self.teff_col = teff_col
        self.lum_col = lum_col
        
        # Weighting options
        self.current_weight_func = "Exponential"
        self.weight_dist = weight_dist
        self.maha_artist = None
        self.corner_fig = None
        self.mass_hist_bw = None
        self.bkg_thin_amount = bkg_thin_amount

        # Histogram attributes
        self.hist_artists = {"m": None, "Myr": None}
        self.track_artists = []
        
        # Dragging state
        self.drag = {"target": None}

        # -----------------------------
        # Figure and axes
        # -----------------------------
        self.fig = plt.figure(figsize=(12, 10))
        gs = self.fig.add_gridspec(2, 2, hspace=0.3)

        self.ax = self.fig.add_subplot(gs[0, :])
        self.hist_axes = {
            "m": self.fig.add_subplot(gs[1, 0]),
            "Myr": self.fig.add_subplot(gs[1, 1]),
        }
        
        self.fig.subplots_adjust(top=0.7)
        self.ax.set_xlim(*self.default_xlim)
        self.ax.set_ylim(*self.default_ylim)
        self.ax.set_yscale("log")
        self.ax.set_xlabel(r"T$_{\rm eff}$ (K)")
        self.ax.set_ylabel(r"L/L$_\odot$")

        self.hist_axes["m"].set(xlabel=r"Mass (M$_\odot$)", ylabel="Probability Density", yticklabels=[])
        self.hist_axes["Myr"].set(xlabel="Age (Myr)", yticklabels=[])


        self.background_scatter()

        # -----------------------------
        # Draggable elements
        # -----------------------------
        self.point, = self.ax.plot([], [], 'o', color='red', markersize=8,
                                   markeredgecolor='black', zorder=3)
        self.hline, = self.ax.plot([], [], color='red', linewidth=2, zorder=1)
        self.vline, = self.ax.plot([], [], color='red', linewidth=2, zorder=1)
        self.hx_l, self.hx_r = [self.ax.plot([], [], 's', color='red', markersize=8,
                                             markeredgecolor='black', picker=5, zorder=2)[0] for _ in range(2)]
        self.hy_b, self.hy_t = [self.ax.plot([], [], 's', color='red', markersize=8,
                                             markeredgecolor='black', picker=5, zorder=2)[0] for _ in range(2)]

        # -----------------------------
        # Initialize controls
        # -----------------------------
        self.boxes = {}
        self.teff_check = None
        self.lum_check = None
        self.init_controls()

        # -----------------------------
        # Connect events
        # -----------------------------
        self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        # -----------------------------
        # Initial plot update
        # -----------------------------
        self.update_plot()
        self.compute()

    # -----------------------------
    # Helper methods
    # -----------------------------
    def compute_positions(self):
        x, y = self.state["Teff"], self.state["Lum"]
        return dict(
            x_left=x - self.state["e_Teff-"],
            x_right=x + self.state["e_Teff+"],
            y_bottom=y - self.state["e_Lum-"],
            y_top=y + self.state["e_Lum+"]
        )

    def update_fields(self):
        self.boxes["Teff"].set_val(f"{self.state['Teff']:.0f}")
        self.boxes["e_Teff+"].set_val(f"{self.state['e_Teff+']:.0f}")
        self.boxes["e_Teff-"].set_val(f"{self.state['e_Teff-']:.0f}")
        self.boxes["Lum"].set_val(f"{self.state['Lum']:.3f}")
        self.boxes["e_Lum+"].set_val(f"{self.state['e_Lum+']:.3f}")
        self.boxes["e_Lum-"].set_val(f"{self.state['e_Lum-']:.3f}")

    def update_plot(self):
        self.validate_position()
        pos = self.compute_positions()
        x, y = self.state["Teff"], self.state["Lum"]
        self.hline.set_data([pos["x_left"], pos["x_right"]], [y, y])
        self.vline.set_data([x, x], [pos["y_bottom"], pos["y_top"]])
        self.point.set_data([x], [y])
        self.hx_l.set_data([pos["x_left"]], [y])
        self.hx_r.set_data([pos["x_right"]], [y])
        self.hy_b.set_data([x], [pos["y_bottom"]])
        self.hy_t.set_data([x], [pos["y_top"]])
        self.pan_if_needed()
        self.update_mahalanobis_fill(max_distance=self.weight_dist)
        self.update_fields()
        self.plot_tracks()
        

    def plot_tracks(self):
        for artist in self.track_artists:
            artist.remove()
        self.track_artists = []

        # -----------------------------
        # Evolutionary tracks
        # -----------------------------
        #plot solar z tracks
        for this_m in np.arange(1.4, 2.5, 0.2):
            this_m = np.round(this_m, 1)
            this_df = self.solar_tracks.query(f"m=={this_m}").sort_values(by='Myr')
            track_line, = self.ax.plot(this_df['teff'].values, this_df['lum'].values, c='k', lw=1.5, zorder=-2)
            self.track_artists.append(track_line)

            ylims = self.ax.get_ylim()
            xlims = self.ax.get_xlim()
            # if the ZAMS (x, y) values are within the plot, label the mass at the ZAMS.
            if (xlims[1] < this_df.teff.values[0] < xlims[0]) and (ylims[0] < this_df.lum.values[0] < ylims[1]):
                track_label = self.ax.text(this_df.teff.values[0], this_df.lum.values[0], f"{this_m}", va='top', ha='center', 
                                           fontsize='small', path_effects=[pe.withStroke(linewidth=5, foreground='white')], zorder=-2)
                self.track_artists.append(track_label)
            # if the TAMS (x, y) values are within the plot, label the mass at the TAMS.
            if (xlims[1] < this_df.teff.values[-1] < xlims[0]) and (ylims[0] < this_df.lum.values[-1] < ylims[1]):
                track_label = self.ax.text(this_df.teff.values[-1], this_df.lum.values[-1], f"{this_m}", va='bottom', ha='left', 
                                           fontsize='small', path_effects=[pe.withStroke(linewidth=5, foreground='white')], zorder=-2)
                self.track_artists.append(track_label)



    def validate_position(self):
        """ Ensure coordinates are not outside the desired data range.
        """
        self.state["Teff"] = max(3500, min(13000, self.state["Teff"]))
        self.state["Lum"] = max(1.5, min(200, self.state["Lum"]))

    # -----------------------------
    # Dynamic panning
    # -----------------------------
    def pan_if_needed(self):
        padding_frac = 0.05
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        xr, yr = x1-x0, y1-y0
        pos = self.compute_positions()
        all_x = [pos["x_left"], pos["x_right"], self.state["Teff"]]
        all_y = [pos["y_bottom"], pos["y_top"], self.state["Lum"]]
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)

        # Horizontal
        if self.ax.get_xscale() == "log":
            left_pad = x0 * (x1/x0)**padding_frac
            right_pad = x1 / (x1/x0)**padding_frac
            if x_max > left_pad:
                factor = x_max / left_pad
                self.ax.set_xlim(x0*factor, x1*factor)
            elif x_min < right_pad:
                factor = x_min / right_pad
                self.ax.set_xlim(x0*factor, x1*factor)
        else:
            dx = 0
            left_pad, right_pad = x0 + xr*padding_frac, x1 - xr*padding_frac
            if x_max > left_pad: 
                dx = x_max - left_pad
            elif x_min < right_pad: 
                dx = x_min - right_pad
            if dx != 0: 
                self.ax.set_xlim(x0+dx, x1+dx)

        # Vertical
        if self.ax.get_yscale() == "log":
            bottom_pad = y0 * (y1/y0)**padding_frac
            top_pad = y1 / (y1/y0)**padding_frac
            if y_min < bottom_pad:
                factor = y_min / bottom_pad
                self.ax.set_ylim(y0*factor, y1*factor)
            elif y_max > top_pad:
                factor = y_max / top_pad
                self.ax.set_ylim(y0*factor, y1*factor)
        else:
            dy = 0
            bottom_pad, top_pad = y0 + yr*padding_frac, y1 - yr*padding_frac
            if y_min < bottom_pad:
                dy = y_min - bottom_pad
            elif y_max > top_pad: 
                dy = y_max - top_pad
            if dy != 0: 
                self.ax.set_ylim(y0+dy, y1+dy)

    # -----------------------------
    # Mahalanobis / weighted fill
    # -----------------------------
    def update_mahalanobis_fill(self, resolution=100, max_distance=None):
        if max_distance is None:
            max_distance = self.weight_dist

        x0, y0 = self.state["Teff"], self.state["Lum"]

        ex_p = self.state["e_Teff+"]
        ex_m = self.state["e_Teff-"]
        ey_p = self.state["e_Lum+"]
        ey_m = self.state["e_Lum-"]

        x_min = x0 - ex_m * max_distance
        x_max = x0 + ex_p * max_distance
        y_min = y0 - ey_m * max_distance
        y_max = y0 + ey_p * max_distance

        x = np.linspace(x_min, x_max, resolution)
        y = np.linspace(y_min, y_max, resolution)

        # broadcasting instead of meshgrid
        dx = (x - x0)[None, :]
        dy = (y - y0)[:, None]

        sigma_x = np.where(x >= x0, ex_p, ex_m)[None, :]
        sigma_y = np.where(y >= y0, ey_p, ey_m)[:, None]
        
        dist = np.sqrt((dx/sigma_x)**2 + (dy/sigma_y)**2)
        W = self.compute_weights(dist, max_distance) 

        extent = [x_min, x_max, y_min, y_max]

        if self.maha_artist is None:
            self.maha_artist = self.ax.imshow(
                W,
                origin="lower",
                extent=extent,
                cmap="viridis_r",
                interpolation="bilinear",
                alpha=0.9,
                zorder=0,
                aspect="auto",
            )
        else:
            self.maha_artist.set_data(W)
            self.maha_artist.set_extent(extent)

        self.fig.canvas.draw_idle()

    def compute_weights(self, distance, max_distance=3):
        D = np.maximum(distance, 1e-12) # avoid blowups
    
        W = np.full_like(D, np.nan)
        mask = D <= max_distance
    
        if self.current_weight_func == "Inverse square":
            W[mask] = 0.25 / (D[mask])**2
            inner = mask & (D <= 0.25)
            W[inner] = 1.0
        else: # mahalanobis exponential
            W[mask] = np.exp(-0.5 * (D[mask])**2)
    
        return W
    # -----------------------------
    # Event handlers
    # -----------------------------
    def pick(self, event):
        for obj, name in [(self.point,"point"),
                          (self.hx_l,"hx_l"),(self.hx_r,"hx_r"),
                          (self.hy_b,"hy_b"),(self.hy_t,"hy_t")]:
            contains, _ = obj.contains(event)
            if contains:
                return name
        return None

    def on_press(self, event):
        if event.inaxes != self.ax: return
        self.drag["target"] = self.pick(event)

    def on_release(self, event):
        if self.drag["target"] is not None:
            self.update_fields()
        self.drag["target"] = None

    def on_motion(self, event):
        if self.drag["target"] is None or event.inaxes != self.ax: return
        x, y = self.state["Teff"], self.state["Lum"]
        if self.drag["target"] == "point":
            self.state["Teff"] = event.xdata
            self.state["Lum"] = event.ydata
        elif self.drag["target"] == "hx_l":
            self.state["e_Teff-"] = max(0, x-event.xdata)
        elif self.drag["target"] == "hx_r":
            self.state["e_Teff+"] = max(0, event.xdata-x)
        elif self.drag["target"] == "hy_b":
            self.state["e_Lum-"] = max(0, y-event.ydata)
        elif self.drag["target"] == "hy_t":
            self.state["e_Lum+"] = max(0, event.ydata-y)
        self.update_plot()

    def on_key(self, event):
        if event.key is None: return
        if hasattr(event,"inaxes") and event.inaxes != self.ax: return
        teff_step, lum_step = 10, 0.01
        if 'shift' in event.key:
            teff_step *= 5
            lum_step *= 5
        if event.key.endswith('left'):
            self.state["Teff"] += teff_step
        elif event.key.endswith('right'):
            self.state["Teff"] -= teff_step
        elif event.key.endswith('up'):
            self.state["Lum"] *= 10**(lum_step)
        elif event.key.endswith('down'):
            self.state["Lum"] /= 10**(lum_step)
        self.update_plot()

    # -----------------------------
    # Top panel controls + Reset
    # -----------------------------
    def init_controls(self):
        layout_text = [
            (r"T$_{\rm eff}$", r"e_T$_{\rm eff}$+", r"e_T$_{\rm eff}$-"),
            ("L", "e_L+", "e_L-"),
        ]
        layout_cols = [
            ("Teff","e_Teff+","e_Teff-"),
            ("Lum","e_Lum+","e_Lum-"),
        ]
        
        box_w, box_h, h_gap, v_gap = 0.10, 0.03, 0.10, 0.01
        ypos = 1.0 - (box_h+v_gap), 1.0 - 2*(box_h+v_gap)
        x_start = self.ax.get_position().x0
        
        for row_col, row_text, y in zip(layout_cols, layout_text, ypos):
            x_cursor = x_start + 0.045
            for i, name in enumerate(row_col):
                self.fig.text(x_cursor, y, row_text[i], ha="left", va="center")
                axbox = plt.axes([x_cursor+0.08, y-0.0125, box_w, box_h])
                box = TextBox(axbox, "", initial=f"{self.state[name]:.6f}")
                self.boxes[name] = box
                box.on_submit(self.make_submit(name))
                x_cursor += box_w + h_gap

        y_bottom = ypos[1] - 0.013
        axbtn = plt.axes([x_cursor, y_bottom, 0.08, 2*box_h+v_gap])
        self.btn_reset = Button(axbtn, "Reset")
        self.btn_reset.on_clicked(self.reset)  # class method
        x_cursor += 0.1

        # -----------------------------
        # Third row widgets (Mass histogram BW + log checkboxes + weight radio buttons + compute button + export button)
        # -----------------------------
        x_cursor = x_start - 0.03
        y_bottom = ypos[1] - 2*(box_h+v_gap) - 0.03

        y_cursor_now = y_bottom
        self.mass_hist_bw = None
        axbox = plt.axes([x_cursor, y_cursor_now, 0.12, 2*box_h])
        axbox.set_title('Mass Hist BW', fontsize='medium', pad=0.52)
        self.bw_box = TextBox(axbox, "", initial="", textalignment='center')
        self.bw_box.on_submit(self.update_mass_hist_bw)

        x_cursor += 0.14
        # log Teff checkbox (top)
        ax_check_teff = plt.axes([x_cursor, y_bottom+box_h, 0.08, box_h])
        ax_check_teff.set_title("HRD scale", fontsize='medium', pad=0.52)
        self.teff_check = CheckButtons(ax_check_teff, [r"$\log$ T$_{\rm eff}$"], [False])
        self.teff_check.on_clicked(self.toggle_teff_log)
        
        # log Lum checkbox (below)
        ax_check_lum = plt.axes([x_cursor, y_bottom, 0.08, box_h])
        self.lum_check = CheckButtons(ax_check_lum, [r"$\log$ L"], [True])
        self.lum_check.on_clicked(self.toggle_lum_log)
        
        # Weight function radio buttons
        x_cursor += 0.10
        ax_radio = plt.axes([x_cursor, y_bottom, 0.18, 2*box_h])
        ax_radio.set_title("Mahalanobis weights", fontsize='medium', pad=0.52)
        self.radio_weight = RadioButtons(ax_radio, ["Exponential", "Inverse square"], active=0)
        self.radio_weight.on_clicked(self.select_weight_function)

        # Include binaries?
        x_cursor += 0.20
        ax_inc_bin = plt.axes([x_cursor, y_bottom, 0.1, 2*box_h])
        self.include_binaries = CheckButtons(ax_inc_bin, ["Include\nBinaries"], [True])
        self.include_binaries.on_clicked(self.toggle_binaries)

        # Compute button
        x_cursor += 0.12
        ax_compute = plt.axes([x_cursor, y_bottom, 0.08, 2*box_h])
        self.btn_compute = Button(ax_compute, "Compute", color="plum", hovercolor="thistle")
        self.btn_compute.on_clicked(self.compute)

        x_cursor += 0.1
        ax_compute = plt.axes([x_cursor, y_bottom+box_h, 0.17, box_h])
        self.btn_export = Button(ax_compute, "Export results", color="plum", hovercolor="thistle")
        self.btn_export.on_clicked(self.export_results)

        y_cursor_now = y_bottom
        self.export_filename = 'results.csv'
        axbox = plt.axes([x_cursor, y_cursor_now, 0.17, box_h])
        self.exportbox = TextBox(axbox, "", initial=f"{self.export_filename}", textalignment='center')
        self.exportbox.on_submit(self.export_name_submit)

        # -----------------------------
        # Fourth row widgets (Corner checkboxes + corner plot button + corner filename box)
        # -----------------------------
        x_cursor = x_start-0.02
        y_bottom = ypos[1] - 4*(box_h+v_gap) - 0.03

        # Checkboxes
        col_names = {'m':'M', 'z':'Z', 'Myr':'Age', 'v_rot':r'V$_{\rm eq}$', 'sini':'sin i', 'P_rot':r'P$_{\rm rot}$', 
                     'R_p':r'R$_{\rm p}$', 'R_eq':r'R$_{\rm eq}$', 'rho':r'$\rho$', self.teff_col:r'T$_{\rm eff}$', self.lum_col:'L', 
                     'new_Dnu':r'$\Delta \nu$', 'p1':'p1', 'p5':'p5'}
        self.corner_default_checks = ['m', 'z', 'Myr', 'v_rot', 'sini', 'P_rot', self.teff_col, self.lum_col]

        self.corner_cols = []
        checkboxes_ax = {}
        self.checkboxes = {}
        for i, (col, col_name) in enumerate(col_names.items()):
            if (i+1)%2 == 0:
                y_cursor_now = y_bottom
            else:
                y_cursor_now = y_bottom+box_h
            checkboxes_ax[col] = plt.axes([x_cursor, y_cursor_now, 0.08, box_h])
            check = True if col in self.corner_default_checks else False
            checkbox = CheckButtons(checkboxes_ax[col], [col_name], [check])
            ## centering
            for text in checkbox.labels:
                text.set_ha("center")
                text.set_va("center")
                # text.set_position((0.5, text.get_position()[1]))
                text.set_position((0.6, 0.45))
            self.checkboxes[col] = checkbox
            checkbox.on_clicked(self.make_corner_checkbox_submit(col))
            if check:
                self.corner_cols.append(col)
            if (i+1)%2 == 0 and i>0:
                x_cursor += 0.09
        self.corner_cols = list(set(self.corner_cols)) # uniqueness

        # make corner button
        x_cursor += 0.01
        ax_corner = plt.axes([x_cursor, y_bottom+box_h, 0.17, box_h])
        self.btn_corner = Button(ax_corner, "Make corner", color="cornflowerblue", hovercolor="thistle")
        self.btn_corner.on_clicked(self.plot_corner)

        # corner filename box
        y_cursor_now = y_bottom
        self.corner_plot_name = 'corner.pdf'
        # self.fig.text(x_cursor, y_bottom, 'Corner savefig', ha="left", va="center")
        axbox = plt.axes([x_cursor, y_cursor_now, 0.17, box_h])
        self.cornerbox = TextBox(axbox, "", initial=f"{self.corner_plot_name}", textalignment='center')
        self.cornerbox.on_submit(self.corner_submit)

    # -----------------------------
    # Control callbacks
    # -----------------------------
    def reset(self, event=None):
        self.state.update(self.default_state)
        self.ax.set_xlim(*self.default_xlim)
        self.ax.set_ylim(*self.default_ylim)
        if self.teff_check.get_status()[0]:
            self.teff_check.set_active(0)
        if not self.lum_check.get_status()[0]:
            self.lum_check.set_active(0)
        if not self.include_binaries.get_status()[0]:
            self.lum_check.set_active(0)

        ## restore corner checkbox states to defaults
        for col, checkbox in self.checkboxes.items():
            is_checked = checkbox.get_status()[0]
            should_be_checked = col in self.corner_default_checks
            if is_checked != should_be_checked:
                checkbox.set_active(0)
        ## sync corner_cols with current states
        self.corner_cols = [col for col, checkbox in self.checkboxes.items() if checkbox.get_status()[0]]

        self.ax.set_xscale("linear")
        self.ax.set_yscale("log")
        self.update_plot()
        self.update_mass_hist_bw(None)
        self.compute()
    
    def toggle_teff_log(self, label):
        self.ax.set_xscale("log" if self.teff_check.get_status()[0] else "linear")
        self.ax.set_xlim(self.default_xlim)
        self.update_plot()
    
    def toggle_lum_log(self, label):
        self.ax.set_yscale("log" if self.lum_check.get_status()[0] else "linear")
        self.ax.set_ylim(self.default_ylim)
        self.update_plot()
    
    def select_weight_function(self, label):
        self.current_weight_func = label
        self.update_mahalanobis_fill()

    def compute(self, event=None):
        x0, y0 = self.state["Teff"], self.state["Lum"]
        ex_p, ex_m = self.state["e_Teff+"], self.state["e_Teff-"]
        ey_p, ey_m = self.state["e_Lum+"], self.state["e_Lum-"]
    
        dx = self.data[self.teff_col] - x0
        dy = self.data[self.lum_col] - y0
    
        sx = np.where(dx >= 0, ex_p, ex_m)
        sy = np.where(dy >= 0, ey_p, ey_m)
    
        dist = np.sqrt((dx/sx)**2 + (dy/sy)**2)
        weights = self.compute_weights(dist, max_distance=3)
        self.data['dist'] = dist
        self.data['weight'] = weights
        weights_mask = np.isfinite(weights)

        self.result = self.data[weights_mask]
        self.result = self.result[self.result['dist'] <= 3]

        self._draw_hist("m", weights)
        self._draw_hist("Myr", weights)
        self.fig.canvas.draw_idle()
        self.hist_axes["m"].set_ybound(lower=0) # This has to occur after `draw_idle`
        self.hist_axes["Myr"].set_ybound(lower=0)

    def _draw_hist(self, label, weights):
        # define a list of x-values that we will evaluate our KDE at
        # we use the full df rather than the points within 3 sigma because we want to minimise edge effects on the KDE
        weights_mask = np.isfinite(weights)
        
        data_col = self.data[label].values[weights_mask]
        
        # calculate a KDE using all points within 3 sigma of the target
        # ACTUALLY USE ALL THE DATA, CHECK THIS TO MAKE SURE IT'S OK
        if label == 'm':
            data_eval_points = np.linspace(data_col.min(), data_col.max(), 132)
            data_col_kde_sig3 = sp.stats.gaussian_kde(data_col, weights=weights[weights_mask], bw_method=self.mass_hist_bw)
            self.mass_hist_bw =  data_col_kde_sig3.factor
            self.bw_box.set_val(np.round(self.mass_hist_bw, 2))
        else:
            data_eval_points = np.arange(data_col.min(), data_col.max(), 3)
            data_col_kde_sig3 = sp.stats.gaussian_kde(data_col, weights=weights[weights_mask])
        data_col_y_kde_sig3 = data_col_kde_sig3.pdf(data_eval_points)
        
        # Define a function to find the root of (CDF - 0.5)
        # The CDF is obtained by integrating the PDF
        def cdf_minus_half_sig3(x):
            return data_col_kde_sig3.integrate_box_1d(-np.inf, x) - 0.5

        # Find the median using fsolve (numerical root-finding)
        # Provide an initial guess for the median (e.g., the sample median)
        data_col_median_kde_sig3 = fsolve(cdf_minus_half_sig3, data_eval_points[np.argmax(data_col_y_kde_sig3)])[0]
        # mode (the x value at which the KDE reaches maximum)
        data_col_mode_kde_sig3 = data_eval_points[np.argmax(data_col_y_kde_sig3)]

        if label == 'm':
            # Estimate uncertainty
            def cdf_percentiles_lower_sig(x):
                return data_col_kde_sig3.integrate_box_1d(-np.inf, x) - 0.5 + 0.341 # the 0.341 gets the percentile corresponding to 1 sigma
            def cdf_percentiles_upper_sig(x):
                return data_col_kde_sig3.integrate_box_1d(-np.inf, x) - 0.5 - 0.341

            sig_lower = fsolve(cdf_percentiles_lower_sig, data_col_mode_kde_sig3)[0]
            sig_upper = fsolve(cdf_percentiles_upper_sig, data_col_mode_kde_sig3)[0]

            up_unc_mass = np.round(sig_upper-data_col_median_kde_sig3,2)
            low_unc_mass = np.round(data_col_median_kde_sig3-sig_lower,2)
        elif label == 'Myr':
            
            # Relative to the mode
            def cdf_mode_lower_sigma(x):
                return data_col_kde_sig3.integrate_box_1d(-np.inf, x) - data_col_kde_sig3.integrate_box_1d(-np.inf, data_col_mode_kde_sig3) + 0.341
            def cdf_mode_upper_sigma(x):
                return data_col_kde_sig3.integrate_box_1d(-np.inf, x) - data_col_kde_sig3.integrate_box_1d(-np.inf, data_col_mode_kde_sig3) - 0.341

            ## hide the stuck iteration warning, edge case. ideally bracketed root solver is to be used. we workaround.
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                sig_lower = fsolve(cdf_mode_lower_sigma, data_col_mode_kde_sig3)[0]
                sig_upper = fsolve(cdf_mode_upper_sigma, data_col_mode_kde_sig3)[0]
            if sig_lower < data_col.min():
                sig_lower = data_col.min()
            elif sig_upper > data_col.max():
                sig_upper = data_col.max()
            up_unc_mass = np.round(sig_upper-data_col_mode_kde_sig3,2)
            low_unc_mass = np.round(data_col_mode_kde_sig3-sig_lower,2)
        
        mass_inds = np.where((sig_lower<data_eval_points) & (data_eval_points<sig_upper))[0]

        if self.hist_artists[label] is None:
            (artist,) = self.hist_axes[label].plot(data_eval_points, data_col_y_kde_sig3)
            self.hist_artists[label] = artist
        else:
            self.hist_artists[label].set_data(data_eval_points, data_col_y_kde_sig3)
            self.hist_artists[label + "_fill"].remove()
            self.hist_artists[label + "_vline"].remove()
            self.hist_artists[label + "_text"].remove()
            
        self.hist_artists[label + "_fill"] = self.hist_axes[label].fill_between(
            data_eval_points[mass_inds],
            data_col_y_kde_sig3[mass_inds],
            y2=0, 
            color='grey', 
            alpha=0.2
        )
        
        sig_value = data_col_median_kde_sig3 if label == 'm' else data_col_mode_kde_sig3
        self.hist_artists[label + "_vline"] = self.hist_axes[label].axvline(
            sig_value,
            c='k',
            ls='dashed',
            alpha=0.8
        )

        if label == "m":
            text = (
                rf"$M = {sig_value:.2f}"
                rf"^{{+{up_unc_mass:.2f}}}"
                rf"_{{-{low_unc_mass:.2f}}}"
                r"\,\mathrm{M_{\odot}}$"
            )
        elif label == "Myr":
            ## Make the age plot to be max 5*sigma age
            self.hist_axes[label].set_xlim(0, sig_value+5*up_unc_mass)
            text = (
                rf"Age$= {sig_value:.0f}"
                rf"^{{+{up_unc_mass:.0f}}}"
                rf"_{{-{low_unc_mass:.0f}}}\,$"
                r"Myr"
            )            
        
        text_x_position = sig_value * 1.05 if label == 'm' else sig_value * 1.3
        current_xlims = self.hist_axes[label].get_xlim()
        if text_x_position+(current_xlims[1]-current_xlims[0])/2.5 > current_xlims[1]:
            text_x_position = current_xlims[0] + 0.05 if label == 'm' else current_xlims[0] + 30

        self.hist_artists[label + "_text"] = self.hist_axes[label].text(
            text_x_position,
            0.9 * max(data_col_y_kde_sig3),
            text,
            fontsize='small',
            path_effects=[pe.withStroke(linewidth=5, foreground='white')]
        )

        self.hist_axes[label].relim()
        self.hist_axes[label].autoscale_view()
        
    def plot_corner(self, event=None):
        self.compute()
        labels_dict = {'m':'M', 'z':'Z', 'Myr':'Age', 'v_rot':r'V$_{\rm eq}$', 'sini':'sin i', 'P_rot':r'P$_{\rm rot}$', 
                     'R_p':r'R$_{\rm p}$', 'R_eq':r'R$_{\rm eq}$', 'rho':r'$\rho$', self.teff_col:r'T$_{\rm eff}$', self.lum_col:r'L', 
                     'new_Dnu':r'$\Delta \nu$', 'p1':'p1', 'p5':'p5'}

        df = self.result.copy()
        weights = df.weight
        all_corner_cols = ['m', 'z', 'Myr', 'v_rot', 'sini', 'P_rot', 'R_p',
            'R_eq', 'rho', self.teff_col, self.lum_col, 'new_Dnu', 'p1', 'p5',]
        selected_cols = [col for col in all_corner_cols if col in self.corner_cols]
        mode_cols = ['Myr', 'sini', 'P_rot']
        selected_mode_cols = [col for col in mode_cols if col in self.corner_cols]

        samples = df[selected_cols].to_numpy()
        ranges = list(np.ones(len(selected_cols)))
        if "P_rot" in selected_cols:
            ranges[selected_cols.index("P_rot")] = 0.90
        labels = [labels_dict[col] for col in selected_cols]
        self.corner_fig = corner.corner(samples, labels=labels, range=ranges, weights=weights.values, color='cornflowerblue', show_titles=False, quantiles=[0.16,0.5,0.84])

        ## title formatting
        title_fmts = np.repeat(".2f", len(selected_cols))
        fmt_overrides = {"z": ".3f", "Myr": ".1f", "v_rot": ".1f", self.teff_col: ".1f"}
        for i, col in enumerate(selected_cols):
            if col in fmt_overrides:
                title_fmts[i] = fmt_overrides[col]

        axes = np.array(self.corner_fig.axes).reshape((len(selected_cols), len(selected_cols)))
        for i, col in enumerate(selected_cols):
            q16, q50, q84 = corner.quantile(samples[:, i], [0.16, 0.5, 0.84], weights=weights.values)
            fmt = title_fmts[i]
            qm, qp = q50-q16, q84-q50
            title = f"${q50:{fmt}}^{{+{qp:{fmt}}}}_{{-{qm:{fmt}}}}$"
            axes[i, i].set_title(title)
        self.corner_fig.savefig(self.corner_plot_name)

    def make_submit(self, key):
        def submit(text):
            try:
                self.state[key] = float(text)
                self.update_plot()
            except ValueError:
                pass
        return submit

    def make_corner_checkbox_submit(self, col):
        def submit(label):
            checked = self.checkboxes[col].get_status()[0]
            if checked and col not in self.corner_cols:
                self.corner_cols.append(col)
            elif not checked and col in self.corner_cols:
                self.corner_cols.remove(col)
            self.corner_cols = list(set(self.corner_cols)) # uniqueness
            self.update_plot()
        return submit

    def corner_submit(self, text):
        self.corner_plot_name = str(text)
    
    def export_results(self, text):
        self.compute()
        self.result.to_csv(self.export_filename, index=False)
        
    def export_name_submit(self, text):
        self.export_filename = str(text)
    
    def update_mass_hist_bw(self, text):
        # text = text.strip()
        if text == "" or text is None:
            self.mass_hist_bw = None
        else:
            self.mass_hist_bw = float(text)

    def background_scatter(self):
        # -----------------------------
        # Background scatter
        # -----------------------------
        if hasattr(self, "background") and self.background is not None:
            self.background.remove()
        self.background = self.ax.scatter(
                        self.data[self.teff_col].values[::self.bkg_thin_amount],
                        self.data[self.lum_col].values[::self.bkg_thin_amount],
                        c='k', alpha=0.2, s=2, zorder=-11)

    def toggle_binaries(self, label):
        if self.include_binaries.get_status()[0]:
            self.data = df_binaries
            self.teff_col = 'bin_teff'
            self.lum_col = 'bin_lum'
            self.background_scatter()
        else:
            self.data = df_no_binaries
            self.teff_col = 'inc_teff'
            self.lum_col = 'inc_lum'
            self.background_scatter()
        self.update_plot()
        self.compute()

    
        
if __name__ == "__main__":
    from pathlib import Path
    from pandas import read_csv, read_feather

    data_dir = Path(__file__).resolve().parent / "data"

    df_binaries = read_feather(data_dir / "full.feather")
    df_no_binaries = read_feather(data_dir / "no_binaries.feather")
    solar_tracks = read_feather(data_dir / "solar_tracks.feather") 

    # df_binaries = read_csv(data_dir / "popsynth_HRD_outputs_extended_apr2026.csv")
    # df_no_binaries = read_csv(data_dir / "popsynth_HRD_outputs_no-binaries_extended_apr2026.csv")
    # solar_tracks = read_csv(data_dir / "solar_tracks.csv")
    
    default_state = {
        "Teff": 8980, "Lum": 15.0,
        "e_Teff+": 180, "e_Teff-": 180,
        "e_Lum+": 3.0, "e_Lum-": 3.0
    }
    default_xlim = (10000, 6250)
    default_ylim = (4, 60)
    weight_dist = 3.0

    plot = InteractiveHRD(df_binaries, solar_tracks, default_state, default_xlim, default_ylim, weight_dist)
    plt.show()
