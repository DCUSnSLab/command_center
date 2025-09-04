#!/usr/bin/env python3
"""
Map Visualizer Tool with Behavior Parameter Editor
PyQt-based tool for visualizing map data and editing node behavior types
"""

import sys
import json
import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import os

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QGraphicsView, QGraphicsScene, QGraphicsItem, QGraphicsEllipseItem,
    QGraphicsLineItem, QGraphicsPathItem, QGraphicsTextItem, QGraphicsPixmapItem, QToolBar,
    QAction, QFileDialog, QMessageBox, QDockWidget, QListWidget, 
    QListWidgetItem, QSplitter, QTreeWidget, QTreeWidgetItem, QComboBox,
    QSpinBox, QDoubleSpinBox, QLineEdit, QTextEdit, QPushButton, QGroupBox,
    QFormLayout, QLabel, QSlider, QCheckBox, QTabWidget, QScrollArea,
    QProgressBar, QStatusBar, QFrame, QDialog, QDialogButtonBox
)
from PyQt5.QtCore import Qt, QPointF, QRectF, pyqtSignal, QTimer, QThread, pyqtSlot, QUrl
from PyQt5.QtGui import (
    QPen, QBrush, QColor, QPainter, QFont, QPainterPath, QPixmap,
    QIcon, QKeySequence, QWheelEvent, QMouseEvent, QPaintEvent, QResizeEvent
)
from PyQt5.QtNetwork import QNetworkAccessManager, QNetworkRequest, QNetworkReply

import urllib.request
import math as Math
from io import BytesIO


@dataclass
class NodeInfo:
    """Node information data class"""
    id: str
    node_type: int
    lat: float
    lon: float
    utm_easting: float
    utm_northing: float
    remark: str = ""
    
    def to_dict(self) -> Dict:
        return asdict(self)


class SatelliteMapTile(QGraphicsPixmapItem):
    """Satellite map tile graphics item"""
    
    def __init__(self, tile_x: int, tile_y: int, zoom: int, tile_size: int = 256):
        super().__init__()
        self.tile_x = tile_x
        self.tile_y = tile_y
        self.zoom = zoom
        self.tile_size = tile_size
        
        # Calculate tile position in scene coordinates
        self.setPos(tile_x * tile_size, tile_y * tile_size)
        
        # Set z-value to be at the bottom
        self.setZValue(-100)
        
        # Load tile image
        self.load_tile_image()
    
    def load_tile_image(self):
        """Load satellite tile image from OpenStreetMap or Google Maps"""
        try:
            # Using OpenStreetMap tiles (free)
            # Alternative: Google Maps, Bing Maps (requires API key)
            url = f"https://tile.openstreetmap.org/{self.zoom}/{self.tile_x}/{self.tile_y}.png"
            
            # For satellite imagery, we can use ESRI World Imagery (free)
            satellite_url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{self.zoom}/{self.tile_y}/{self.tile_x}"
            
            # Try to load the image
            response = urllib.request.urlopen(satellite_url, timeout=5)
            image_data = response.read()
            
            pixmap = QPixmap()
            pixmap.loadFromData(image_data)
            
            if not pixmap.isNull():
                # Scale tile to proper size
                scaled_pixmap = pixmap.scaled(self.tile_size, self.tile_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.setPixmap(scaled_pixmap)
            else:
                # Create placeholder tile
                self.create_placeholder_tile()
                
        except Exception as e:
            # Create placeholder tile if loading fails
            self.create_placeholder_tile()
    
    def create_placeholder_tile(self):
        """Create a placeholder tile when satellite image fails to load"""
        pixmap = QPixmap(self.tile_size, self.tile_size)
        pixmap.fill(QColor(240, 240, 240))  # Light gray background
        
        painter = QPainter(pixmap)
        painter.setPen(QColor(200, 200, 200))
        painter.setFont(QFont("Arial", 8))
        painter.drawText(pixmap.rect(), Qt.AlignCenter, f"Tile\n{self.tile_x},{self.tile_y}\nZ{self.zoom}")
        painter.end()
        
        self.setPixmap(pixmap)


class SatelliteMapManager:
    """Manager for satellite map tiles"""
    
    def __init__(self, scene: QGraphicsScene):
        self.scene = scene
        self.tiles = {}  # Dictionary to store loaded tiles
        self.tile_size = 256
        self.current_zoom = 18  # Good zoom level for detailed view
        self.enable_satellite = True
    
    def deg2num(self, lat_deg: float, lon_deg: float, zoom: int) -> Tuple[int, int]:
        """Convert lat/lon to tile numbers"""
        lat_rad = Math.radians(lat_deg)
        n = 2.0 ** zoom
        xtile = int((lon_deg + 180.0) / 360.0 * n)
        ytile = int((1.0 - Math.asinh(Math.tan(lat_rad)) / Math.pi) / 2.0 * n)
        return xtile, ytile
    
    def num2deg(self, xtile: int, ytile: int, zoom: int) -> Tuple[float, float]:
        """Convert tile numbers to lat/lon"""
        n = 2.0 ** zoom
        lon_deg = xtile / n * 360.0 - 180.0
        lat_rad = Math.atan(Math.sinh(Math.pi * (1 - 2 * ytile / n)))
        lat_deg = Math.degrees(lat_rad)
        return lat_deg, lon_deg
    
    def load_tiles_for_bounds(self, min_lat: float, max_lat: float, min_lon: float, max_lon: float):
        """Load satellite tiles for given bounds"""
        if not self.enable_satellite:
            return
        
        # Convert bounds to tile coordinates
        min_tile_x, max_tile_y = self.deg2num(min_lat, min_lon, self.current_zoom)
        max_tile_x, min_tile_y = self.deg2num(max_lat, max_lon, self.current_zoom)
        
        # Ensure proper ordering
        if min_tile_x > max_tile_x:
            min_tile_x, max_tile_x = max_tile_x, min_tile_x
        if min_tile_y > max_tile_y:
            min_tile_y, max_tile_y = max_tile_y, min_tile_y
        
        # Load tiles in the visible area
        for x in range(min_tile_x, max_tile_x + 1):
            for y in range(min_tile_y, max_tile_y + 1):
                tile_key = f"{x}_{y}_{self.current_zoom}"
                
                if tile_key not in self.tiles:
                    try:
                        # Create and add new tile
                        tile = SatelliteMapTile(x, y, self.current_zoom)
                        self.tiles[tile_key] = tile
                        self.scene.addItem(tile)
                    except Exception as e:
                        print(f"Failed to load tile {x},{y},{self.current_zoom}: {e}")
    
    def clear_tiles(self):
        """Clear all loaded tiles"""
        tiles_to_remove = list(self.tiles.keys())
        for tile_key in tiles_to_remove:
            tile = self.tiles[tile_key]
            try:
                if tile.scene() is not None:
                    self.scene.removeItem(tile)
            except RuntimeError:
                # Tile already deleted
                pass
            del self.tiles[tile_key]
    
    def set_satellite_enabled(self, enabled: bool):
        """Enable or disable satellite imagery"""
        self.enable_satellite = enabled
        
        # Check if tiles still exist before setting visibility
        tiles_to_remove = []
        for tile_key, tile in self.tiles.items():
            try:
                if tile.scene() is not None:  # Check if tile is still in scene
                    tile.setVisible(enabled)
                else:
                    tiles_to_remove.append(tile_key)
            except RuntimeError:
                # Tile has been deleted
                tiles_to_remove.append(tile_key)
        
        # Remove deleted tiles from dictionary
        for tile_key in tiles_to_remove:
            del self.tiles[tile_key]
    
    def set_zoom_level(self, zoom: int):
        """Set zoom level for tiles"""
        if zoom != self.current_zoom:
            self.clear_tiles()
            self.current_zoom = max(1, min(18, zoom))  # Clamp zoom level


@dataclass
class LinkInfo:
    """Link information data class"""
    id: str
    from_node_id: str
    to_node_id: str
    length: float
    remark: str = ""
    
    def to_dict(self) -> Dict:
        return asdict(self)


class MapGraphicsView(QGraphicsView):
    """Custom graphics view with pan and zoom capabilities"""
    
    node_selected = pyqtSignal(str)  # Node ID selected
    node_double_clicked = pyqtSignal(str)  # Node ID double clicked
    
    def __init__(self):
        super().__init__()
        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.RubberBandDrag)
        self.setInteractive(True)
        self.setMouseTracking(True)
        
        # Pan and zoom settings
        self._pan = False
        self._pan_start_x = 0
        self._pan_start_y = 0
        self._zoom_factor = 1.15
        self._min_zoom = 0.1
        self._max_zoom = 10.0
        
        # Current zoom level
        self._current_zoom = 1.0
    
    def wheelEvent(self, event: QWheelEvent):
        """Handle mouse wheel for zooming"""
        # Get the position of the mouse cursor
        old_pos = self.mapToScene(event.pos())
        
        # Zoom
        zoom_in = event.angleDelta().y() > 0
        if zoom_in and self._current_zoom < self._max_zoom:
            self.scale(self._zoom_factor, self._zoom_factor)
            self._current_zoom *= self._zoom_factor
        elif not zoom_in and self._current_zoom > self._min_zoom:
            self.scale(1.0 / self._zoom_factor, 1.0 / self._zoom_factor)
            self._current_zoom /= self._zoom_factor
        
        # Get the new position
        new_pos = self.mapToScene(event.pos())
        
        # Move scene to maintain mouse position
        delta = new_pos - old_pos
        self.translate(delta.x(), delta.y())
    
    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press for panning and selection"""
        if event.button() == Qt.MiddleButton:
            self._pan = True
            self._pan_start_x = event.x()
            self._pan_start_y = event.y()
            self.setCursor(Qt.ClosedHandCursor)
        else:
            super().mousePressEvent(event)
            
            # Check for node selection
            scene_pos = self.mapToScene(event.pos())
            item = self.scene().itemAt(scene_pos, self.transform())
            if item and hasattr(item, 'node_id'):
                self.node_selected.emit(item.node_id)
    
    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse movement for panning"""
        if self._pan:
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - (event.x() - self._pan_start_x)
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - (event.y() - self._pan_start_y)
            )
            self._pan_start_x = event.x()
            self._pan_start_y = event.y()
        else:
            super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release"""
        if event.button() == Qt.MiddleButton:
            self._pan = False
            self.setCursor(Qt.ArrowCursor)
        else:
            super().mouseReleaseEvent(event)
    
    def mouseDoubleClickEvent(self, event: QMouseEvent):
        """Handle double click for node editing"""
        scene_pos = self.mapToScene(event.pos())
        item = self.scene().itemAt(scene_pos, self.transform())
        if item and hasattr(item, 'node_id'):
            self.node_double_clicked.emit(item.node_id)
        else:
            super().mouseDoubleClickEvent(event)
    
    def fit_to_content(self):
        """Fit view to show all content"""
        self.fitInView(self.scene().itemsBoundingRect(), Qt.KeepAspectRatio)
        self._current_zoom = 1.0


class NodeGraphicsItem(QGraphicsEllipseItem):
    """Custom graphics item for map nodes"""
    
    def __init__(self, node: NodeInfo, radius: float = 3.0):
        super().__init__(-radius, -radius, radius * 2, radius * 2)
        self.node_id = node.id
        self.node_info = node
        self.radius = radius
        
        # Set position
        self.setPos(node.utm_easting, -node.utm_northing)  # Flip Y for proper display
        
        # Set appearance based on node type
        self.update_appearance()
        
        # Enable selection and hovering
        self.setFlags(QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemIsFocusable)
        self.setAcceptHoverEvents(True)
    
    def update_appearance(self):
        """Update node appearance based on node type"""
        node_type = self.node_info.node_type
        
        # Behavior type colors
        colors = {
            1: QColor(0, 255, 0),      # Forward - Green
            2: QColor(255, 165, 0),    # Reverse - Orange  
            3: QColor(0, 255, 255),    # Precise Forward - Cyan
            4: QColor(255, 255, 0),    # Precise Reverse - Yellow
            5: QColor(255, 0, 0),      # Fast Forward - Red
            6: QColor(128, 0, 128),    # Slow Forward - Purple
            7: QColor(255, 192, 203),  # Pause 1s - Pink
            8: QColor(255, 20, 147),   # Pause 4s - Deep Pink
            9: QColor(0, 0, 255),      # End Point - Blue
            10: QColor(255, 255, 255), # Traffic Light - White
            11: QColor(128, 128, 128)  # Lane Ignore - Gray
        }
        
        color = colors.get(node_type, QColor(128, 128, 128))
        
        # Set brush and pen
        self.setBrush(QBrush(color))
        self.setPen(QPen(QColor(0, 0, 0), 1))
        
        # Larger radius for special node types
        if node_type in [7, 8, 9, 10]:  # Pause, End, Traffic Light
            new_radius = self.radius * 1.5
            self.setRect(-new_radius, -new_radius, new_radius * 2, new_radius * 2)
    
    def hoverEnterEvent(self, event):
        """Handle mouse hover enter"""
        self.setPen(QPen(QColor(255, 255, 255), 2))
        self.setToolTip(f"Node: {self.node_id}\\nType: {self.node_info.node_type}\\n"
                       f"UTM: ({self.node_info.utm_easting:.1f}, {self.node_info.utm_northing:.1f})")
        super().hoverEnterEvent(event)
    
    def hoverLeaveEvent(self, event):
        """Handle mouse hover leave"""
        self.setPen(QPen(QColor(0, 0, 0), 1))
        super().hoverLeaveEvent(event)


class LinkGraphicsItem(QGraphicsLineItem):
    """Custom graphics item for map links"""
    
    def __init__(self, link: LinkInfo, from_node: NodeInfo, to_node: NodeInfo):
        # Create line from from_node to to_node
        super().__init__(
            from_node.utm_easting, -from_node.utm_northing,
            to_node.utm_easting, -to_node.utm_northing
        )
        
        self.link_id = link.id
        self.link_info = link
        
        # Set appearance
        self.setPen(QPen(QColor(128, 128, 128), 1))
        
        # Set z-value to be behind nodes
        self.setZValue(-1)


class NodeEditDialog(QDialog):
    """Dialog for editing node properties"""
    
    def __init__(self, node: NodeInfo, parent=None):
        super().__init__(parent)
        self.node = node
        self.setWindowTitle(f"Edit Node {node.id}")
        self.setModal(True)
        self.resize(400, 300)
        
        self.setup_ui()
        self.load_node_data()
    
    def setup_ui(self):
        """Setup dialog UI"""
        layout = QVBoxLayout()
        
        # Node info group
        info_group = QGroupBox("Node Information")
        info_layout = QFormLayout()
        
        self.id_edit = QLineEdit()
        self.id_edit.setReadOnly(True)
        info_layout.addRow("ID:", self.id_edit)
        
        # Node type combo
        self.type_combo = QComboBox()
        behavior_types = {
            1: "Forward",
            2: "Reverse", 
            3: "Precise Forward",
            4: "Precise Reverse",
            5: "Fast Forward",
            6: "Slow Forward", 
            7: "Pause 1s",
            8: "Pause 4s",
            9: "End Point",
            10: "Traffic Light",
            11: "Lane Ignore"
        }
        
        for type_id, type_name in behavior_types.items():
            self.type_combo.addItem(f"{type_id}: {type_name}", type_id)
        
        info_layout.addRow("Node Type:", self.type_combo)
        
        # Coordinates (read-only)
        self.lat_edit = QLineEdit()
        self.lat_edit.setReadOnly(True)
        info_layout.addRow("Latitude:", self.lat_edit)
        
        self.lon_edit = QLineEdit()
        self.lon_edit.setReadOnly(True)
        info_layout.addRow("Longitude:", self.lon_edit)
        
        self.utm_e_edit = QLineEdit()
        self.utm_e_edit.setReadOnly(True)
        info_layout.addRow("UTM Easting:", self.utm_e_edit)
        
        self.utm_n_edit = QLineEdit()
        self.utm_n_edit.setReadOnly(True)
        info_layout.addRow("UTM Northing:", self.utm_n_edit)
        
        # Remark
        self.remark_edit = QTextEdit()
        self.remark_edit.setMaximumHeight(80)
        info_layout.addRow("Remark:", self.remark_edit)
        
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
        self.setLayout(layout)
    
    def load_node_data(self):
        """Load node data into form"""
        self.id_edit.setText(self.node.id)
        
        # Set combo box to current node type
        index = self.type_combo.findData(self.node.node_type)
        if index >= 0:
            self.type_combo.setCurrentIndex(index)
        
        self.lat_edit.setText(f"{self.node.lat:.8f}")
        self.lon_edit.setText(f"{self.node.lon:.8f}")
        self.utm_e_edit.setText(f"{self.node.utm_easting:.3f}")
        self.utm_n_edit.setText(f"{self.node.utm_northing:.3f}")
        self.remark_edit.setPlainText(self.node.remark)
    
    def get_updated_node(self) -> NodeInfo:
        """Get updated node information"""
        updated_node = NodeInfo(
            id=self.node.id,
            node_type=self.type_combo.currentData(),
            lat=self.node.lat,
            lon=self.node.lon,
            utm_easting=self.node.utm_easting,
            utm_northing=self.node.utm_northing,
            remark=self.remark_edit.toPlainText()
        )
        return updated_node


class BatchEditDialog(QDialog):
    """Dialog for batch editing multiple nodes"""
    
    def __init__(self, nodes: List[NodeInfo], parent=None):
        super().__init__(parent)
        self.nodes = nodes
        self.setWindowTitle(f"Batch Edit {len(nodes)} Nodes")
        self.setModal(True)
        self.resize(500, 400)
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup dialog UI"""
        layout = QVBoxLayout()
        
        # Info label
        info_label = QLabel(f"Editing {len(self.nodes)} selected nodes")
        info_label.setStyleSheet("font-weight: bold; margin: 10px;")
        layout.addWidget(info_label)
        
        # Node list
        list_group = QGroupBox("Selected Nodes")
        list_layout = QVBoxLayout()
        
        self.node_list = QListWidget()
        self.node_list.setMaximumHeight(150)
        
        for node in self.nodes:
            item_text = f"{node.id} (Type: {node.node_type})"
            self.node_list.addItem(item_text)
        
        list_layout.addWidget(self.node_list)
        list_group.setLayout(list_layout)
        layout.addWidget(list_group)
        
        # Edit options
        edit_group = QGroupBox("Batch Operations")
        edit_layout = QFormLayout()
        
        # Change node type
        self.change_type_cb = QCheckBox("Change Node Type")
        self.type_combo = QComboBox()
        behavior_types = {
            1: "Forward", 2: "Reverse", 3: "Precise Forward", 4: "Precise Reverse",
            5: "Fast Forward", 6: "Slow Forward", 7: "Pause 1s", 8: "Pause 4s",
            9: "End Point", 10: "Traffic Light", 11: "Lane Ignore"
        }
        
        for type_id, type_name in behavior_types.items():
            self.type_combo.addItem(f"{type_id}: {type_name}", type_id)
        
        self.type_combo.setEnabled(False)
        self.change_type_cb.toggled.connect(self.type_combo.setEnabled)
        
        type_layout = QHBoxLayout()
        type_layout.addWidget(self.change_type_cb)
        type_layout.addWidget(self.type_combo)
        edit_layout.addRow("Node Type:", type_layout)
        
        # Add remark
        self.add_remark_cb = QCheckBox("Add/Replace Remark")
        self.remark_edit = QLineEdit()
        self.remark_edit.setEnabled(False)
        self.remark_edit.setPlaceholderText("Enter remark for all selected nodes")
        self.add_remark_cb.toggled.connect(self.remark_edit.setEnabled)
        
        remark_layout = QHBoxLayout()
        remark_layout.addWidget(self.add_remark_cb)
        remark_layout.addWidget(self.remark_edit)
        edit_layout.addRow("Remark:", remark_layout)
        
        edit_group.setLayout(edit_layout)
        layout.addWidget(edit_group)
        
        # Preview
        preview_group = QGroupBox("Preview Changes")
        preview_layout = QVBoxLayout()
        
        self.preview_text = QTextEdit()
        self.preview_text.setMaximumHeight(100)
        self.preview_text.setReadOnly(True)
        preview_layout.addWidget(self.preview_text)
        
        preview_btn = QPushButton("Preview Changes")
        preview_btn.clicked.connect(self.update_preview)
        preview_layout.addWidget(preview_btn)
        
        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)
        
        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
        self.setLayout(layout)
        
        # Initial preview
        self.update_preview()
    
    def update_preview(self):
        """Update preview of changes"""
        preview_text = "Changes to be applied:\n\n"
        
        if self.change_type_cb.isChecked():
            new_type = self.type_combo.currentData()
            type_name = self.type_combo.currentText()
            preview_text += f"• Change node type to: {type_name}\n"
            preview_text += f"  Affected nodes: {len(self.nodes)}\n\n"
        
        if self.add_remark_cb.isChecked():
            remark = self.remark_edit.text().strip()
            if remark:
                preview_text += f"• Set remark to: '{remark}'\n"
                preview_text += f"  Affected nodes: {len(self.nodes)}\n\n"
        
        if not self.change_type_cb.isChecked() and not self.add_remark_cb.isChecked():
            preview_text += "No changes selected."
        
        self.preview_text.setPlainText(preview_text)
    
    def get_updated_nodes(self) -> List[NodeInfo]:
        """Get updated nodes with applied changes"""
        updated_nodes = []
        
        for node in self.nodes:
            # Create a copy of the node
            updated_node = NodeInfo(
                id=node.id,
                node_type=node.node_type,
                lat=node.lat,
                lon=node.lon,
                utm_easting=node.utm_easting,
                utm_northing=node.utm_northing,
                remark=node.remark
            )
            
            # Apply changes
            if self.change_type_cb.isChecked():
                updated_node.node_type = self.type_combo.currentData()
            
            if self.add_remark_cb.isChecked():
                remark = self.remark_edit.text().strip()
                if remark:
                    updated_node.remark = remark
            
            updated_nodes.append(updated_node)
        
        return updated_nodes


class NodeSelectionDialog(QDialog):
    """Dialog for selecting multiple nodes for batch editing"""
    
    def __init__(self, nodes: Dict[str, NodeInfo], parent=None):
        super().__init__(parent)
        self.nodes = nodes
        self.setWindowTitle("Select Nodes for Batch Edit")
        self.setModal(True)
        self.resize(600, 500)
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup dialog UI"""
        layout = QVBoxLayout()
        
        # Filter options
        filter_group = QGroupBox("Filter Nodes")
        filter_layout = QHBoxLayout()
        
        # Filter by node type
        filter_layout.addWidget(QLabel("Node Type:"))
        self.type_filter = QComboBox()
        self.type_filter.addItem("All Types", -1)
        
        behavior_types = {
            1: "Forward", 2: "Reverse", 3: "Precise Forward", 4: "Precise Reverse",
            5: "Fast Forward", 6: "Slow Forward", 7: "Pause 1s", 8: "Pause 4s",
            9: "End Point", 10: "Traffic Light", 11: "Lane Ignore"
        }
        
        for type_id, type_name in behavior_types.items():
            self.type_filter.addItem(f"{type_id}: {type_name}", type_id)
        
        self.type_filter.currentIndexChanged.connect(self.apply_filters)
        filter_layout.addWidget(self.type_filter)
        
        # Search by ID
        filter_layout.addWidget(QLabel("Search ID:"))
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Enter node ID or partial ID")
        self.search_edit.textChanged.connect(self.apply_filters)
        filter_layout.addWidget(self.search_edit)
        
        filter_group.setLayout(filter_layout)
        layout.addWidget(filter_group)
        
        # Node list with checkboxes
        list_group = QGroupBox("Select Nodes")
        list_layout = QVBoxLayout()
        
        # Selection buttons
        select_buttons = QHBoxLayout()
        select_all_btn = QPushButton("Select All")
        select_none_btn = QPushButton("Select None")
        select_visible_btn = QPushButton("Select Visible")
        
        select_all_btn.clicked.connect(self.select_all_nodes)
        select_none_btn.clicked.connect(self.select_no_nodes)
        select_visible_btn.clicked.connect(self.select_visible_nodes)
        
        select_buttons.addWidget(select_all_btn)
        select_buttons.addWidget(select_none_btn)
        select_buttons.addWidget(select_visible_btn)
        select_buttons.addStretch()
        
        list_layout.addLayout(select_buttons)
        
        # Node tree widget
        self.node_tree = QTreeWidget()
        self.node_tree.setHeaderLabels(["Node ID", "Type", "UTM Coordinates", "Remark"])
        self.node_tree.setRootIsDecorated(False)
        self.node_tree.setSelectionMode(QTreeWidget.MultiSelection)
        
        list_layout.addWidget(self.node_tree)
        
        # Selection count
        self.selection_label = QLabel("Selected: 0 nodes")
        list_layout.addWidget(self.selection_label)
        
        list_group.setLayout(list_layout)
        layout.addWidget(list_group)
        
        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
        self.setLayout(layout)
        
        # Populate node tree
        self.populate_node_tree()
        
        # Connect selection change
        self.node_tree.itemSelectionChanged.connect(self.update_selection_count)
    
    def populate_node_tree(self, filtered_nodes=None):
        """Populate the node tree widget"""
        self.node_tree.clear()
        
        nodes_to_show = filtered_nodes if filtered_nodes else self.nodes
        
        for node in nodes_to_show.values():
            item = QTreeWidgetItem([
                node.id,
                str(node.node_type),
                f"({node.utm_easting:.1f}, {node.utm_northing:.1f})",
                node.remark[:50] + "..." if len(node.remark) > 50 else node.remark
            ])
            item.setData(0, Qt.UserRole, node.id)  # Store node ID
            self.node_tree.addTopLevelItem(item)
        
        # Resize columns
        for i in range(self.node_tree.columnCount()):
            self.node_tree.resizeColumnToContents(i)
    
    def apply_filters(self):
        """Apply filters to node list"""
        filtered_nodes = {}
        
        # Filter by type
        selected_type = self.type_filter.currentData()
        
        # Filter by search text
        search_text = self.search_edit.text().strip().upper()
        
        for node_id, node in self.nodes.items():
            # Type filter
            if selected_type != -1 and node.node_type != selected_type:
                continue
            
            # Search filter
            if search_text and search_text not in node.id.upper():
                continue
            
            filtered_nodes[node_id] = node
        
        self.populate_node_tree(filtered_nodes)
        self.update_selection_count()
    
    def select_all_nodes(self):
        """Select all visible nodes"""
        for i in range(self.node_tree.topLevelItemCount()):
            item = self.node_tree.topLevelItem(i)
            item.setSelected(True)
    
    def select_no_nodes(self):
        """Deselect all nodes"""
        self.node_tree.clearSelection()
    
    def select_visible_nodes(self):
        """Select all currently visible nodes"""
        self.select_all_nodes()
    
    def update_selection_count(self):
        """Update selection count label"""
        count = len(self.node_tree.selectedItems())
        self.selection_label.setText(f"Selected: {count} nodes")
    
    def get_selected_nodes(self) -> List[NodeInfo]:
        """Get list of selected nodes"""
        selected_nodes = []
        
        for item in self.node_tree.selectedItems():
            node_id = item.data(0, Qt.UserRole)
            if node_id in self.nodes:
                selected_nodes.append(self.nodes[node_id])
        
        return selected_nodes


class MapVisualizerWindow(QMainWindow):
    """Main window for map visualization and editing"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Map Visualizer & Behavior Parameter Editor")
        self.setGeometry(100, 100, 1400, 900)
        
        # Data storage
        self.nodes: Dict[str, NodeInfo] = {}
        self.links: Dict[str, LinkInfo] = {}
        self.node_items: Dict[str, NodeGraphicsItem] = {}
        self.link_items: Dict[str, LinkGraphicsItem] = {}
        
        # Satellite map manager
        self.satellite_manager = None
        
        # Current file path
        self.current_file_path: Optional[str] = None
        self.is_modified = False
        
        # Setup UI
        self.setup_ui()
        self.setup_menus()
        self.setup_status_bar()
        
        # Load default map if exists
        default_map = os.path.join(os.path.dirname(__file__), 'mando_full_map.json')
        if os.path.exists(default_map):
            self.load_map_file(default_map)
    
    def setup_ui(self):
        """Setup main UI components"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create splitter for main layout
        main_splitter = QSplitter(Qt.Horizontal)
        
        # Map view in center (create this first)
        self.map_view = MapGraphicsView()
        self.map_scene = QGraphicsScene()
        self.map_view.setScene(self.map_scene)
        
        # Initialize satellite map manager
        self.satellite_manager = SatelliteMapManager(self.map_scene)
        
        # Left panel for controls (create after map_view)
        left_panel = self.create_left_panel()
        left_panel.setMaximumWidth(300)
        left_panel.setMinimumWidth(250)
        
        # Connect signals
        self.map_view.node_selected.connect(self.on_node_selected)
        self.map_view.node_double_clicked.connect(self.on_node_double_clicked)
        
        # Add to splitter
        main_splitter.addWidget(left_panel)
        main_splitter.addWidget(self.map_view)
        
        # Set splitter proportions
        main_splitter.setStretchFactor(0, 0)  # Left panel fixed
        main_splitter.setStretchFactor(1, 1)  # Map view expandable
        
        # Set main layout
        layout = QHBoxLayout()
        layout.addWidget(main_splitter)
        central_widget.setLayout(layout)
    
    def create_left_panel(self) -> QWidget:
        """Create left control panel"""
        panel = QWidget()
        layout = QVBoxLayout()
        
        # Map info group
        info_group = QGroupBox("Map Information")
        info_layout = QFormLayout()
        
        self.nodes_count_label = QLabel("0")
        self.links_count_label = QLabel("0")
        info_layout.addRow("Nodes:", self.nodes_count_label)
        info_layout.addRow("Links:", self.links_count_label)
        
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        # Display options group
        display_group = QGroupBox("Display Options")
        display_layout = QVBoxLayout()
        
        self.show_nodes_cb = QCheckBox("Show Nodes")
        self.show_nodes_cb.setChecked(True)
        self.show_nodes_cb.toggled.connect(self.update_display_options)
        
        self.show_links_cb = QCheckBox("Show Links")
        self.show_links_cb.setChecked(True)
        self.show_links_cb.toggled.connect(self.update_display_options)
        
        self.show_node_ids_cb = QCheckBox("Show Node IDs")
        self.show_node_ids_cb.toggled.connect(self.update_display_options)
        
        self.show_satellite_cb = QCheckBox("Show Satellite Map")
        self.show_satellite_cb.setChecked(True)
        self.show_satellite_cb.toggled.connect(self.update_satellite_display)
        
        display_layout.addWidget(self.show_nodes_cb)
        display_layout.addWidget(self.show_links_cb)
        display_layout.addWidget(self.show_node_ids_cb)
        display_layout.addWidget(self.show_satellite_cb)
        
        display_group.setLayout(display_layout)
        layout.addWidget(display_group)
        
        # Node filter group
        filter_group = QGroupBox("Node Type Filter")
        filter_layout = QVBoxLayout()
        
        self.node_type_filters = {}
        behavior_types = {
            1: "Forward", 2: "Reverse", 3: "Precise Forward", 4: "Precise Reverse",
            5: "Fast Forward", 6: "Slow Forward", 7: "Pause 1s", 8: "Pause 4s",
            9: "End Point", 10: "Traffic Light", 11: "Lane Ignore"
        }
        
        for type_id, type_name in behavior_types.items():
            cb = QCheckBox(f"{type_id}: {type_name}")
            cb.setChecked(True)
            cb.toggled.connect(self.update_node_filters)
            self.node_type_filters[type_id] = cb
            filter_layout.addWidget(cb)
        
        filter_group.setLayout(filter_layout)
        
        # Add scroll area for filters
        filter_scroll = QScrollArea()
        filter_scroll.setWidget(filter_group)
        filter_scroll.setWidgetResizable(True)
        filter_scroll.setMaximumHeight(200)
        layout.addWidget(filter_scroll)
        
        # Selected node info
        self.selected_node_group = QGroupBox("Selected Node")
        selected_layout = QFormLayout()
        
        self.selected_id_label = QLabel("None")
        self.selected_type_label = QLabel("None")
        self.selected_coords_label = QLabel("None")
        
        selected_layout.addRow("ID:", self.selected_id_label)
        selected_layout.addRow("Type:", self.selected_type_label)
        selected_layout.addRow("Coordinates:", self.selected_coords_label)
        
        self.edit_node_btn = QPushButton("Edit Node")
        self.edit_node_btn.setEnabled(False)
        self.edit_node_btn.clicked.connect(self.edit_selected_node)
        selected_layout.addWidget(self.edit_node_btn)
        
        self.selected_node_group.setLayout(selected_layout)
        layout.addWidget(self.selected_node_group)
        
        # Control buttons
        buttons_layout = QHBoxLayout()
        
        fit_btn = QPushButton("Fit to View")
        fit_btn.clicked.connect(self.map_view.fit_to_content)
        
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh_map)
        
        buttons_layout.addWidget(fit_btn)
        buttons_layout.addWidget(refresh_btn)
        
        layout.addLayout(buttons_layout)
        layout.addStretch()
        
        panel.setLayout(layout)
        return panel
    
    def setup_menus(self):
        """Setup menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        open_action = QAction("Open Map...", self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.triggered.connect(self.open_map_file)
        file_menu.addAction(open_action)
        
        save_action = QAction("Save", self)
        save_action.setShortcut(QKeySequence.Save)
        save_action.triggered.connect(self.save_map_file)
        file_menu.addAction(save_action)
        
        save_as_action = QAction("Save As...", self)
        save_as_action.setShortcut(QKeySequence.SaveAs)
        save_as_action.triggered.connect(self.save_map_file_as)
        file_menu.addAction(save_as_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.setShortcut(QKeySequence.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # View menu
        view_menu = menubar.addMenu("View")
        
        fit_action = QAction("Fit to View", self)
        fit_action.triggered.connect(self.map_view.fit_to_content)
        view_menu.addAction(fit_action)
        
        # Tools menu
        tools_menu = menubar.addMenu("Tools")
        
        batch_edit_action = QAction("Batch Edit Node Types...", self)
        batch_edit_action.triggered.connect(self.batch_edit_nodes)
        tools_menu.addAction(batch_edit_action)
        
        select_edit_action = QAction("Select and Edit Nodes...", self)
        select_edit_action.triggered.connect(self.select_and_edit_nodes)
        tools_menu.addAction(select_edit_action)
        
        validate_action = QAction("Validate Map Data", self)
        validate_action.triggered.connect(self.validate_map_data)
        tools_menu.addAction(validate_action)
    
    def setup_status_bar(self):
        """Setup status bar"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)
        
        self.status_bar.showMessage("Ready")
    
    def load_map_file(self, file_path: str):
        """Load map data from JSON file"""
        try:
            self.status_bar.showMessage(f"Loading map file: {file_path}")
            self.progress_bar.setVisible(True)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Clear existing data
            self.nodes.clear()
            self.links.clear()
            self.map_scene.clear()
            self.node_items.clear()
            self.link_items.clear()
            
            # Load nodes
            if 'Node' in data:
                total_nodes = len(data['Node'])
                for i, node_data in enumerate(data['Node']):
                    if i % 100 == 0:  # Update progress every 100 nodes
                        self.progress_bar.setValue(int(i * 50 / total_nodes))
                        QApplication.processEvents()
                    
                    node = NodeInfo(
                        id=node_data['ID'],
                        node_type=node_data.get('NodeType', 1),
                        lat=node_data['GpsInfo']['Lat'],
                        lon=node_data['GpsInfo']['Long'],
                        utm_easting=node_data['UtmInfo']['Easting'],
                        utm_northing=node_data['UtmInfo']['Northing'],
                        remark=node_data.get('Remark', '')
                    )
                    self.nodes[node.id] = node
            
            # Load links
            if 'Link' in data:
                total_links = len(data['Link'])
                for i, link_data in enumerate(data['Link']):
                    if i % 100 == 0:  # Update progress
                        self.progress_bar.setValue(50 + int(i * 50 / total_links))
                        QApplication.processEvents()
                    
                    link = LinkInfo(
                        id=link_data['ID'],
                        from_node_id=link_data['FromNodeID'],
                        to_node_id=link_data['ToNodeID'],
                        length=link_data.get('Length', 0.0),
                        remark=link_data.get('Remark', '')
                    )
                    self.links[link.id] = link
            
            self.current_file_path = file_path
            self.is_modified = False
            
            # Update UI
            self.update_map_info()
            self.render_map()
            
            # Load satellite tiles for the map bounds
            self.load_satellite_tiles()
            
            self.map_view.fit_to_content()
            
            self.progress_bar.setVisible(False)
            self.status_bar.showMessage(f"Loaded {len(self.nodes)} nodes, {len(self.links)} links")
            
        except Exception as e:
            self.progress_bar.setVisible(False)
            QMessageBox.critical(self, "Error", f"Failed to load map file: {str(e)}")
            self.status_bar.showMessage("Failed to load map file")
    
    def render_map(self):
        """Render map on graphics scene"""
        self.map_scene.clear()
        self.node_items.clear()
        self.link_items.clear()
        
        # Render links first (so they appear behind nodes)
        if self.show_links_cb.isChecked():
            for link in self.links.values():
                if link.from_node_id in self.nodes and link.to_node_id in self.nodes:
                    from_node = self.nodes[link.from_node_id]
                    to_node = self.nodes[link.to_node_id]
                    
                    link_item = LinkGraphicsItem(link, from_node, to_node)
                    self.map_scene.addItem(link_item)
                    self.link_items[link.id] = link_item
        
        # Render nodes
        if self.show_nodes_cb.isChecked():
            for node in self.nodes.values():
                # Check if this node type should be displayed
                if node.node_type in self.node_type_filters:
                    if self.node_type_filters[node.node_type].isChecked():
                        node_item = NodeGraphicsItem(node)
                        self.map_scene.addItem(node_item)
                        self.node_items[node.id] = node_item
        
        # Update scene bounds
        self.map_scene.setSceneRect(self.map_scene.itemsBoundingRect())
    
    def load_satellite_tiles(self):
        """Load satellite tiles for current map bounds"""
        if not self.nodes or not self.satellite_manager:
            return
        
        # Calculate map bounds from nodes
        lats = [node.lat for node in self.nodes.values()]
        lons = [node.lon for node in self.nodes.values()]
        
        if lats and lons:
            min_lat, max_lat = min(lats), max(lats)
            min_lon, max_lon = min(lons), max(lons)
            
            # Add some padding
            lat_padding = (max_lat - min_lat) * 0.1
            lon_padding = (max_lon - min_lon) * 0.1
            
            self.satellite_manager.load_tiles_for_bounds(
                min_lat - lat_padding, max_lat + lat_padding,
                min_lon - lon_padding, max_lon + lon_padding
            )
    
    def update_satellite_display(self):
        """Update satellite map display"""
        if self.satellite_manager:
            self.satellite_manager.set_satellite_enabled(self.show_satellite_cb.isChecked())
    
    def update_map_info(self):
        """Update map information display"""
        self.nodes_count_label.setText(str(len(self.nodes)))
        self.links_count_label.setText(str(len(self.links)))
    
    def update_display_options(self):
        """Update display options"""
        self.render_map()
    
    def update_node_filters(self):
        """Update node type filters"""
        self.render_map()
    
    @pyqtSlot(str)
    def on_node_selected(self, node_id: str):
        """Handle node selection"""
        if node_id in self.nodes:
            node = self.nodes[node_id]
            
            self.selected_id_label.setText(node.id)
            self.selected_type_label.setText(f"{node.node_type}")
            self.selected_coords_label.setText(f"({node.utm_easting:.1f}, {node.utm_northing:.1f})")
            self.edit_node_btn.setEnabled(True)
            
            self.current_selected_node_id = node_id
    
    @pyqtSlot(str)
    def on_node_double_clicked(self, node_id: str):
        """Handle node double click for editing"""
        self.current_selected_node_id = node_id
        self.edit_selected_node()
    
    def edit_selected_node(self):
        """Edit the currently selected node"""
        if not hasattr(self, 'current_selected_node_id'):
            return
        
        node_id = self.current_selected_node_id
        if node_id not in self.nodes:
            return
        
        dialog = NodeEditDialog(self.nodes[node_id], self)
        if dialog.exec_() == QDialog.Accepted:
            # Update node data
            updated_node = dialog.get_updated_node()
            self.nodes[node_id] = updated_node
            
            # Update graphics item
            if node_id in self.node_items:
                self.node_items[node_id].node_info = updated_node
                self.node_items[node_id].update_appearance()
            
            # Update selection display
            self.on_node_selected(node_id)
            
            self.is_modified = True
            self.status_bar.showMessage(f"Updated node {node_id}")
    
    def refresh_map(self):
        """Refresh map display"""
        self.render_map()
        self.status_bar.showMessage("Map refreshed")
    
    def open_map_file(self):
        """Open map file dialog"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Map File", "", "JSON Files (*.json);;All Files (*)"
        )
        
        if file_path:
            self.load_map_file(file_path)
    
    def save_map_file(self):
        """Save current map file"""
        if self.current_file_path:
            self.save_to_file(self.current_file_path)
        else:
            self.save_map_file_as()
    
    def save_map_file_as(self):
        """Save map file as dialog"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Map File", "", "JSON Files (*.json);;All Files (*)"
        )
        
        if file_path:
            self.save_to_file(file_path)
            self.current_file_path = file_path
    
    def save_to_file(self, file_path: str):
        """Save map data to file"""
        try:
            # Convert back to original JSON format
            data = {
                "Node": [],
                "Link": []
            }
            
            # Convert nodes
            for node in self.nodes.values():
                node_data = {
                    "ID": node.id,
                    "AdminCode": "110",
                    "NodeType": node.node_type,
                    "ITSNodeID": f"ITS_{node.id}",
                    "Maker": "한국도로공사",
                    "UpdateDate": "20250418",
                    "Version": "2021",
                    "Remark": node.remark,
                    "HistType": "02A",
                    "HistRemark": "순차 경로 노드",
                    "GpsInfo": {
                        "Lat": node.lat,
                        "Long": node.lon,
                        "Alt": 0.0
                    },
                    "UtmInfo": {
                        "Easting": node.utm_easting,
                        "Northing": node.utm_northing,
                        "Zone": "52N"
                    }
                }
                data["Node"].append(node_data)
            
            # Convert links
            for link in self.links.values():
                link_data = {
                    "ID": link.id,
                    "AdminCode": "110",
                    "RoadRank": 1,
                    "RoadType": 1,
                    "RoadNo": "20",
                    "LinkType": 3,
                    "LaneNo": 2,
                    "R_LinkID": f"R_{link.id[1:]}",
                    "L_LinkID": f"L_{link.id[1:]}",
                    "FromNodeID": link.from_node_id,
                    "ToNodeID": link.to_node_id,
                    "SectionID": "SEQ_SECTION_00",
                    "Length": link.length,
                    "ITSLinkID": f"ITS_{link.id}",
                    "Maker": "한국도로공사",
                    "UpdateDate": "20250418",
                    "Version": "2021",
                    "Remark": link.remark,
                    "HistType": "02A"
                }
                data["Link"].append(link_data)
            
            # Save to file
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            self.is_modified = False
            self.status_bar.showMessage(f"Saved to {file_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save map file: {str(e)}")
    
    def batch_edit_nodes(self):
        """Batch edit node types for all nodes of selected type"""
        if not self.nodes:
            QMessageBox.information(self, "Info", "No nodes loaded.")
            return
        
        # Get unique node types
        node_types = set(node.node_type for node in self.nodes.values())
        
        if not node_types:
            return
        
        # Ask user which type to edit
        behavior_types = {
            1: "Forward", 2: "Reverse", 3: "Precise Forward", 4: "Precise Reverse",
            5: "Fast Forward", 6: "Slow Forward", 7: "Pause 1s", 8: "Pause 4s",
            9: "End Point", 10: "Traffic Light", 11: "Lane Ignore"
        }
        
        # Create selection dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Node Type to Edit")
        layout = QVBoxLayout()
        
        layout.addWidget(QLabel("Select which node type to batch edit:"))
        
        type_combo = QComboBox()
        for node_type in sorted(node_types):
            type_name = behavior_types.get(node_type, f"Unknown ({node_type})")
            count = sum(1 for node in self.nodes.values() if node.node_type == node_type)
            type_combo.addItem(f"{node_type}: {type_name} ({count} nodes)", node_type)
        
        layout.addWidget(type_combo)
        
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        
        dialog.setLayout(layout)
        
        if dialog.exec_() == QDialog.Accepted:
            selected_type = type_combo.currentData()
            nodes_to_edit = [node for node in self.nodes.values() if node.node_type == selected_type]
            
            if nodes_to_edit:
                batch_dialog = BatchEditDialog(nodes_to_edit, self)
                if batch_dialog.exec_() == QDialog.Accepted:
                    self.apply_batch_edit(batch_dialog.get_updated_nodes())
    
    def select_and_edit_nodes(self):
        """Select specific nodes and edit them"""
        if not self.nodes:
            QMessageBox.information(self, "Info", "No nodes loaded.")
            return
        
        # Open node selection dialog
        selection_dialog = NodeSelectionDialog(self.nodes, self)
        if selection_dialog.exec_() == QDialog.Accepted:
            selected_nodes = selection_dialog.get_selected_nodes()
            
            if selected_nodes:
                batch_dialog = BatchEditDialog(selected_nodes, self)
                if batch_dialog.exec_() == QDialog.Accepted:
                    self.apply_batch_edit(batch_dialog.get_updated_nodes())
            else:
                QMessageBox.information(self, "Info", "No nodes selected.")
    
    def apply_batch_edit(self, updated_nodes: List[NodeInfo]):
        """Apply batch edit changes to nodes"""
        updated_count = 0
        
        for updated_node in updated_nodes:
            if updated_node.id in self.nodes:
                # Update node data
                self.nodes[updated_node.id] = updated_node
                
                # Update graphics item
                if updated_node.id in self.node_items:
                    self.node_items[updated_node.id].node_info = updated_node
                    self.node_items[updated_node.id].update_appearance()
                
                updated_count += 1
        
        self.is_modified = True
        self.status_bar.showMessage(f"Updated {updated_count} nodes")
        
        # Refresh display
        self.render_map()
    
    def validate_map_data(self):
        """Validate map data for consistency"""
        # TODO: Implement validation logic
        QMessageBox.information(self, "Info", "Map validation feature coming soon!")


def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    app.setApplicationName("Map Visualizer")
    app.setApplicationVersion("1.0.0")
    
    # Set application icon (if available)
    app.setWindowIcon(QIcon())
    
    window = MapVisualizerWindow()
    window.show()
    
    return app.exec_()


if __name__ == '__main__':
    sys.exit(main())