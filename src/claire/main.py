import sys

from PySide6.QtWidgets import QMainWindow, QApplication

from claire.app import App


def main() -> None:
    app = QApplication(sys.argv)

    window = QMainWindow()
    window.setCentralWidget(App())
    window.resize(800, 600)
    window.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
