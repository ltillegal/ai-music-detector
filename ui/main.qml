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

    property string verdict: ""
    property string summary: ""

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
            window.verdict = obj.verdict || ""
            var art = obj.p_ai_artifactnet
            var son = obj.p_ai_sonics
            var parts = []
            if (art !== undefined) parts.push("ArtifactNet " + Number(art).toFixed(3))
            if (son !== undefined) parts.push("SONICS " + Number(son).toFixed(3))
            window.summary = parts.join("   ·   ")
            resultTextView.color = obj.verdict === "AI"
                ? "#ff1744"
                : obj.verdict === "Human"
                    ? "#00e676"
                    : obj.verdict === "Error"
                        ? "#ff9100"
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

        Label {
            visible: window.verdict.length > 0
            text: window.verdict
            font.pixelSize: 28
            font.bold: true
            color: window.verdict === "AI"
                ? "#ff1744"
                : window.verdict === "Human"
                    ? "#00e676"
                    : "#ffd740"
        }

        Label {
            visible: window.summary.length > 0
            text: window.summary
            opacity: 0.8
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