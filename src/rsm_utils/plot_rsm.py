import os
import sys
import numpy as np
import glob
import pyvista as pv
import pyqtgraph as pg
import matplotlib.pyplot as plt
from pyvistaqt import QtInteractor

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton,
    QVBoxLayout, QWidget, QFileDialog, 
    QLabel, QLineEdit, QComboBox, QHBoxLayout
)
from PyQt5.QtGui import QColor

class VTIViewer(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("3D RSM Viewer")
        self.setGeometry(100, 100, 1200, 600)

        # ---- State ----
        self.grid = None
        self.scalar_name = None
        self.current_dir = '/home/beams22/29IDUSER/Documents/User_Macros'
        self.cmap_lst = ["viridis", "plasma", "inferno", "magma", "jet"]
        self.cmap_str = self.cmap_lst[0]
        # self.rsm_path = os.path.dirname(__file__)
        # self.cmap_lst = []
        # cmap_flist = glob.glob(os.path.join(self.rsm_path, 'cmaps', '*.npy'))
        # for cmap_fpath in cmap_flist:
        #     cmap_name = cmap_fpath.split('/')[-1]
        #     self.cmap_lst.append(cmap_name.split('.')[0])

        # ---- UI Layout ----
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout()
        central.setLayout(main_layout)

        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()
        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)

        # ---- 3D View ----
        self.plotter = QtInteractor(self)
        self.plotter.set_background('white')
        left_layout.addWidget(self.plotter.interactor)

        # ---- Controls Row ----
        control_layout = QHBoxLayout()
        left_layout.addLayout(control_layout)

        # Load button
        self.load_btn = QPushButton("Load VTK")
        self.load_btn.clicked.connect(self.load_vtk)
        control_layout.addWidget(self.load_btn)

        # Contour input
        self.level_input = QLineEdit()
        self.level_input.setPlaceholderText("Contour (e.g. 1e-3)")
        self.level_input.returnPressed.connect(self.update_contour)
        control_layout.addWidget(self.level_input)

        # Colormap dropdown
        self.cmap_box = QComboBox()
        self.cmap_box.addItems(self.cmap_lst)
        self.cmap_box.currentTextChanged.connect(self.update_plots)
        control_layout.addWidget(self.cmap_box)

        # Info label
        self.label = QLabel("No data loaded")
        left_layout.addWidget(self.label)

        # --- 2D slice panel (right) ---
        slice_layout = QVBoxLayout()
        right_layout.addLayout(slice_layout, 1)  # 1x width

        self.slice_view = pg.PlotWidget()
        self.pcm = pg.PColorMeshItem()
        self.slice_view.addItem(self.pcm)
        self.slice_view.showGrid(x=True, y=True, alpha=0.3)
        self.slice_view.getViewBox().setAspectLocked(True, ratio=1)
        slice_layout.addWidget(self.slice_view)

        slice_control_layout = QHBoxLayout()
        slice_layout.addLayout(slice_control_layout)
        self.slice_input = QLineEdit("L=0.95")
        slice_control_layout.addWidget(self.slice_input)
        slice_control_layout.addWidget(QLabel('vmin:'))
        self.vmin_input = QLineEdit(str(0))
        slice_control_layout.addWidget(self.vmin_input)
        slice_control_layout.addWidget(QLabel('vmax:'))
        self.vmax_input = QLineEdit(str(1))
        slice_control_layout.addWidget(self.vmax_input)
        self.slice_button = QPushButton("Plot Slice")
        
        slice_control_layout.addWidget(self.slice_button)

        self.slice_button.clicked.connect(self.update_slice)
        self.vmin_input.returnPressed.connect(self.update_vmin_vmax)
        self.vmax_input.returnPressed.connect(self.update_vmin_vmax)

    # -------------------------
    # Load VTI / VTS
    # -------------------------
    def load_vtk(self):
        self.log_norm = False

        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Open VTK File",
            self.current_dir,
            "VTK Files (*.vti *.vts)"
        )
        if not file_name:
            return

        self.current_dir = os.path.dirname(file_name)

        self.grid = pv.read(file_name)

        # detect scalar
        if self.grid.point_data:
            self.scalar_name = list(self.grid.point_data.keys())[0]
        elif self.grid.cell_data:
            self.scalar_name = list(self.grid.cell_data.keys())[0]
        else:
            raise ValueError("No scalar data found")

        # default contour (log midpoint)
        data = self.grid[self.scalar_name]
        data = np.clip(data, 1e-12, None)
        nx, ny, nz = self.grid.dimensions
        self.H_array = self.grid.x
        self.K_array = self.grid.y
        self.L_array = self.grid.z
        self.data_array = data.reshape(nx, ny, nz, order='F')

        vmin = data.min()
        vmax = data.max()
        default_level = 10**((np.log10(vmin) + np.log10(vmax)) / 2)

        self.level_input.setText(f"{default_level:.2e}")

        # setup plot
        self.plotter.clear()
        self.plotter.reset_camera(bounds=self.grid.bounds)
        
        self.plotter.add_axes(
            xlabel="H",
            ylabel="K",
            zlabel="L",
            labels_off=False,
        )

        self.plotter.show_bounds(
            bounds=self.grid.bounds,
            location="outer",
            all_edges=True,
        )

        self.plotter.set_scale(1, 1, 1)
        self.update_contour()

    # -------------------------
    # Update contour
    # -------------------------
    def update_plots(self):
        self.update_contour()
        self.update_slice()

    def update_contour(self):
        if self.grid is None:
            return

        data = self.grid[self.scalar_name]
        data = np.clip(data, 1e-12, None)

        vmin = data.min()
        vmax = data.max()

        # read input
        try:
            levels = [float(x) for x in self.level_input.text().split(",")]
            opacity_levels = [0.05,]*len(levels)
            self.cmap_str = self.cmap_box.currentText()
            self.plotter.clear()
            for level, opacity in zip(levels, opacity_levels):
                contour = self.grid.contour([level], scalars=self.scalar_name)
                self.plotter.add_mesh(
                    contour,
                    scalars=self.scalar_name,
                    cmap=self.cmap_str,
                    opacity=opacity,
                    # log_scale=True,
                    reset_camera=False
                )
        except Exception:
            levels = [120, 150, 200]
            self.cmap_str = self.cmap_box.currentText()
            self.plotter.clear()
            for level, opacity in zip(levels, [0.05] * len(levels)):
                contour = self.grid.contour([level], scalars=self.scalar_name)
                self.plotter.add_mesh(
                    contour,
                    scalars=self.scalar_name,
                    cmap=self.cmap_str,
                    opacity=opacity,
                    # log_scale=True,
                    reset_camera=False
                )
        # self.plotter.show_grid()
        cubeaxesactor = self.plotter.show_grid(font_size=10)
        cubeaxesactor.x_label_format='{:.3f}'
        cubeaxesactor.y_label_format='{:.3f}'
        cubeaxesactor.z_label_format='{:.3f}'
        cubeaxesactor.x_title='H'
        cubeaxesactor.y_title='K'
        cubeaxesactor.z_title='L'

        # self.plotter.reset_camera(bounds=self.grid.bounds)
        
        self.label.setText(f"Contours: {levels}")
        self.plotter.render()

    def update_slice(self):
        """Plot 2D slice based on input like 'L=25'"""
        slice_str = self.slice_input.text().strip()
        axis_map = {'H': 0, 'K': 1, 'L': 2}

        try:
            axis_char, val = slice_str.split('=')
            axis_char, val = axis_char.strip(), val.strip()
            axis_char = axis_char.upper()
            val = float(val)

            if axis_char not in axis_map:
                raise ValueError("Axis must be H, K, or L")

            axis = axis_map[axis_char]

            # Map value to nearest voxel index
            min_val = self.grid.bounds[axis*2]
            max_val = self.grid.bounds[axis*2+1]
            idx = int(round((val - min_val)/(max_val - min_val) * (self.data_array.shape[axis]-1)))

            # Extract slice
            if axis == 0:
                slc = np.s_[idx, :, :]
                x_grid = self.K_array[slc]
                y_grid = self.L_array[slc]
                slice_2d = self.data_array[idx, :-1, :-1]
                xlabel = 'K'
                ylabel = 'L'
            elif axis == 1:
                slc = np.s_[:, idx, :]
                x_grid = self.H_array[slc]
                y_grid = self.L_array[slc]
                slice_2d = self.data_array[:-1, idx, :-1]
                xlabel = 'H'
                ylabel = 'L'
            else:
                slc = np.s_[:, :, idx]
                x_grid = self.H_array[slc]
                y_grid = self.K_array[slc]
                slice_2d = self.data_array[:-1, :-1, idx]
                xlabel = 'H'
                ylabel = 'K'

            
            # cmap = pg.colormap.get(self.cmap_str)
            # lut = cmap.getLookupTable(0.0, 1.0, 256)
            # if lut.shape[1] == 3:
            #     alpha = np.ones((lut.shape[0], 1), dtype=lut.dtype)*255
            #     lut = np.hstack((lut, alpha))
            cmap = plt.get_cmap(self.cmap_str)
            lut = cmap(np.linspace(0, 1, 256))
            lut = (lut * 255).astype(np.uint8)
            lut_colors = [QColor(r, g, b, a) for r, g, b, a in lut]
            self.pcm.setLookupTable(lut_colors)
            if self.log_norm:
                self.pcm.setData(x_grid, y_grid, np.log(slice_2d))
            else:
                self.pcm.setData(x_grid, y_grid, slice_2d)
            self.slice_view.setXRange(x_grid.min(), x_grid.max())
            self.slice_view.setYRange(y_grid.min(), y_grid.max())
            self.slice_view.setLabel('bottom', xlabel)
            self.slice_view.setLabel('left', ylabel)
            self.update_vmin_vmax()

        except Exception as e:
            print("Error plotting slice:", e)

    def update_vmin_vmax(self):
        if self.pcm:
            self.slice_vmin = float(self.vmin_input.text())
            self.slice_vmax = float(self.vmax_input.text())
            if self.slice_vmin < self.slice_vmax:
                if self.log_norm:
                    self.pcm.setLevels([np.log(self.slice_vmin), np.log(self.slice_vmax)])
                else:
                    self.pcm.setLevels([self.slice_vmin, self.slice_vmax])

# -------------------------
# Run App
# -------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VTIViewer()
    window.show()
    sys.exit(app.exec_())