import logging
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QComboBox, QTableWidget, QTableWidgetItem, QHeaderView,
    QMessageBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QBrush

from .base_window import BaseWindow


class MeasurementHistoryWindow(BaseWindow):
    def __init__(self, user_role=None):
        super().__init__(show_back_button=True, show_power_off=False)
        self.user_role = user_role
        self.logger = logging.getLogger(__name__)
        self.init_ui()
        self.load_patients()

    def init_ui(self):
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)

        # Title
        title_label = QLabel('Measurement History')
        title_label.setStyleSheet(f"""
            font-size: {self.scaled(24)}px;
            font-weight: bold;
            color: #2c3e50;
            margin: {self.scaled(20)}px 0;
        """)
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)

        # Patient section (scoped style so it doesn't affect children)
        patient_section = QWidget()
        patient_section.setObjectName("patientSection")
        patient_section.setStyleSheet(f"""
            #patientSection {{
                background-color: transparent;
                border-radius: {self.scaled(10)}px;
                padding: {self.scaled(15)}px;
                margin: {self.scaled(10)}px 0;
            }}
        """)
        patient_layout = QHBoxLayout(patient_section)

        patient_label = QLabel('Select Patient:')
        patient_label.setStyleSheet(f"""
            font-weight: bold;
            font-size: {self.scaled(16)}px;
            color: #2c3e50;
        """)
        patient_layout.addWidget(patient_label)

        self.patient_combo = QComboBox()
        self.patient_combo.addItem('Select Patient...')
        self.patient_combo.currentIndexChanged.connect(self.on_patient_selected)
        self.patient_combo.setStyleSheet(f"""
            QComboBox {{
                background-color: white;
                border: 2px solid #D0D9E7;
                border-radius: {self.scaled(5)}px;
                padding: {self.scaled(8)}px;
                min-height: {self.scaled(35)}px;
                font-size: {self.scaled(14)}px;
            }}
            QComboBox::drop-down {{ border: none; width: {self.scaled(20)}px; }}
            QComboBox::down-arrow {{
                image: none;
                border-left: {self.scaled(5)}px solid transparent;
                border-right: {self.scaled(5)}px solid transparent;
                border-top: {self.scaled(5)}px solid #666;
                margin-right: {self.scaled(5)}px;
            }}
        """)
        patient_layout.addWidget(self.patient_combo)

        self.refresh_btn = QPushButton('Refresh')
        self.refresh_btn.clicked.connect(self.load_patients)
        self.refresh_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: #28a745;
                color: white;
                border: none;
                border-radius: {self.scaled(5)}px;
                padding: {self.scaled(8)}px {self.scaled(15)}px;
                font-size: {self.scaled(12)}px;
                min-height: {self.scaled(35)}px;
                font-weight: bold;
            }}
            QPushButton:hover {{ background-color: #218838; }}
        """)
        patient_layout.addWidget(self.refresh_btn)
        main_layout.addWidget(patient_section)

        # Table section (scoped style)
        table_section = QWidget()
        table_section.setObjectName("tableSection")
        table_section.setStyleSheet(f"""
            #tableSection {{
                background-color: transparent;
                border-radius: {self.scaled(10)}px;
                padding: {self.scaled(15)}px;
                margin: {self.scaled(10)}px 0;
            }}
        """)
        table_layout = QVBoxLayout(table_section)

        table_title = QLabel('Measurement History')
        table_title.setStyleSheet(f"""
            font-weight: bold;
            font-size: {self.scaled(16)}px;
            color: #2c3e50;
            margin-bottom: {self.scaled(10)}px;
        """)
        table_layout.addWidget(table_title)

        # ---- NATIVE TABLE + HEADER (no custom header widgets) ----
        self.measurement_table = QTableWidget()
        self.measurement_table.setColumnCount(7)  # 6 visible + 1 hidden (ID)
        self.measurement_table.setHorizontalHeaderLabels(
            ['Date', 'Time', 'Avg Heart Rate', 'Status', 'Duration (s)', 'Measurements', 'ID']
        )

        # Always show native header
        self.measurement_table.horizontalHeader().setVisible(True)

        # Table visuals
        self.measurement_table.setStyleSheet(f"""
            QTableWidget {{
                background-color: white;
                border: none;
                border-radius: {self.scaled(5)}px;
                gridline-color: #2c3e50;
                selection-background-color: #4A90E2;
                selection-color: white;
            }}
            QTableWidget::item {{
                padding: {self.scaled(8)}px;
                border-bottom: 1px solid #E0E0E0;
            }}
        """)

        # Header styling (no per-section vertical borders -> no misalignment)
        header = self.measurement_table.horizontalHeader()
        header.setStyleSheet(f"""
            QHeaderView {{
                border-top-left-radius: {self.scaled(5)}px;
                border-top-right-radius: {self.scaled(5)}px;
            }}
            QHeaderView::section {{
                background-color: #4A90E2;
                color: white;
                padding: {self.scaled(8)}px;
                font-weight: bold;
                font-size: {self.scaled(12)}px;
                border: none;
                border-bottom: 1px solid #2c3e50; /* aligns with grid below */
                min-height: {self.scaled(32)}px;
            }}
            QTableCornerButton::section {{
                background-color: #4A90E2;
                border: none;
                border-bottom: 1px solid #2c3e50;
            }}
        """)
        header.setDefaultAlignment(Qt.AlignCenter)
        header.setStretchLastSection(False)

        self.measurement_table.verticalHeader().setVisible(False)
        self.measurement_table.setShowGrid(True)
        self.measurement_table.setAlternatingRowColors(True)
        self.measurement_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.measurement_table.setSelectionMode(QTableWidget.SingleSelection)
        self.measurement_table.setSortingEnabled(True)
        self.measurement_table.setCornerButtonEnabled(False)
        self.measurement_table.setFocusPolicy(Qt.NoFocus)

        # Exact widths (match your plan)
        widths = [120, 170, 140, 100, 120, 120, 0]  # last is hidden ID
        for i, w in enumerate(widths):
            header.setSectionResizeMode(i, QHeaderView.Fixed)
            header.resizeSection(i, self.scaled(w))

        self.measurement_table.setColumnHidden(6, True)

        # Size & scrollbars
        content_width = sum(widths[:-1])
        self.measurement_table.setFixedWidth(self.scaled(content_width))
        self.measurement_table.setFixedHeight(self.scaled(400))
        self.measurement_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.measurement_table.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        table_layout.addWidget(self.measurement_table, 0, Qt.AlignLeft)
        main_layout.addWidget(table_section)
        main_layout.addStretch()
        self.content_layout.addWidget(main_widget)

    # ---------------- Data loading ----------------

    def load_patients(self):
        try:
            from database.db import Database
            db = Database()
            user_role = getattr(self, 'user_role', None)
            patients = db.get_all_patients() if user_role == 'Administrator' else db.get_all_patients()

            self.patient_combo.currentIndexChanged.disconnect()
            self.patient_combo.clear()
            self.patient_combo.addItem('Select Patient...')
            for p in patients:
                self.patient_combo.addItem(f"{p['firstName']} {p['lastName']} (ID: {p['id']})", p['id'])
            self.patient_combo.currentIndexChanged.connect(self.on_patient_selected)
            self.logger.info(f"Loaded {len(patients)} patients into combo box")

        except Exception as e:
            self.logger.error(f"Error loading patients: {str(e)}")
            self.patient_combo.clear()
            self.patient_combo.addItem('Select Patient...')
            self.patient_combo.addItem('Error loading patients')
            try:
                self.patient_combo.currentIndexChanged.connect(self.on_patient_selected)
            except:
                pass

    def get_selected_patient_id(self):
        idx = self.patient_combo.currentIndex()
        return self.patient_combo.itemData(idx) if idx > 0 else None

    def on_patient_selected(self):
        pid = self.get_selected_patient_id()
        if pid:
            self.load_measurement_history(pid)
        else:
            self.measurement_table.setRowCount(0)

    def load_measurement_history(self, patient_id):
        try:
            from database.db import Database
            db = Database()
            sessions = db.get_measurement_sessions(patient_id)

            self.measurement_table.setRowCount(0)

            if not sessions:
                self.measurement_table.setRowCount(1)
                msg = QTableWidgetItem("No measurement history found for this patient")
                msg.setTextAlignment(Qt.AlignCenter)
                msg.setFlags(Qt.ItemIsEnabled)
                self.measurement_table.setItem(0, 0, msg)
                self.measurement_table.setSpan(0, 0, 1, 6)
                return

            self.measurement_table.setRowCount(len(sessions))

            for row, s in enumerate(sessions):
                d = QTableWidgetItem(str(s['measurement_date'])); d.setTextAlignment(Qt.AlignCenter)
                t = QTableWidgetItem(str(s['measurement_time'])); t.setTextAlignment(Qt.AlignCenter)

                hr = QTableWidgetItem(f"{s['avg_heart_rate']:.1f} BPM")
                hr.setTextAlignment(Qt.AlignCenter)
                status = s.get('status', '')
                if status == 'Normal':
                    hr.setBackground(QBrush(QColor(200, 255, 200)))
                elif status == 'Bradycardia':
                    hr.setBackground(QBrush(QColor(255, 245, 200)))
                elif status == 'Tachycardia':
                    hr.setBackground(QBrush(QColor(255, 210, 210)))

                st = QTableWidgetItem(status); st.setTextAlignment(Qt.AlignCenter)
                du = QTableWidgetItem(f"{s['duration_seconds']:.1f}"); du.setTextAlignment(Qt.AlignCenter)
                me = QTableWidgetItem(str(s['total_measurements'])); me.setTextAlignment(Qt.AlignCenter)
                iid = QTableWidgetItem(str(s['id']))

                self.measurement_table.setItem(row, 0, d)
                self.measurement_table.setItem(row, 1, t)
                self.measurement_table.setItem(row, 2, hr)
                self.measurement_table.setItem(row, 3, st)
                self.measurement_table.setItem(row, 4, du)
                self.measurement_table.setItem(row, 5, me)
                self.measurement_table.setItem(row, 6, iid)

            self.logger.info(f"Loaded {len(sessions)} measurement sessions for patient {patient_id}")

        except Exception as e:
            self.logger.error(f"Error loading measurement history: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to load measurement history: {str(e)}")

    def go_back(self):
        from .home_window import HomeWindow
        self.home_window = HomeWindow(user_role=getattr(self, 'user_role', None))
        self.home_window.show()
        self.close()

    def closeEvent(self, event):
        try:
            event.accept()
        except Exception as e:
            self.logger.error(f"Error in close event: {str(e)}")
            event.accept()
