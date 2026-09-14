import QtQuick
import QtQuick.Controls
import QtQuick.Controls.Material
import QtQuick.Layouts

ApplicationWindow {
    id: window
    width: 560
    height: 360
    visible: true
    title: "Детекция AI-музыки"

    Material.theme: Material.Dark
    Material.accent: Material.Purple
    Material.primary: Material.BlueGrey

    Connections {
        target: bridge
        function onResultChanged(text) {
            resultTextView.text = text
            var obj
            try {
                obj = JSON.parse(text)
            } catch (e) {
                return
            }
            resultTextView.color = obj.verdict === "AI"
                ? "#ff1744"
                : obj.verdict === "Human"
                    ? "#00e676"
                    : "#ffd740"
        }
    }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 16
        spacing: 12

        RowLayout {
            Layout.fillWidth: true
            spacing: 8

            Button {
                text: "Выбрать файл"
                onClicked: bridge.chooseFile()
            }

            Button {
                text: "Запись с микрофона"
                onClicked: bridge.detectMic()
            }
        }

        Label {
            text: "Результат:"
        }

        Flickable {
            id: flick
            Layout.fillWidth: true
            Layout.fillHeight: true
            clip: true
            contentWidth: resultTextView.paintedWidth
            contentHeight: resultTextView.paintedHeight

            TextEdit {
                id: resultTextView
                width: flick.width
                readOnly: true
                selectByMouse: true
                wrapMode: TextEdit.Wrap
                font.family: "monospace"
                text: ""
            }
        }
    }
}