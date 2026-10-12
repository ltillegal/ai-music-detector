import QtQuick
import QtQuick.Controls
import QtQuick.Controls.Material
import QtQuick.Layouts

ApplicationWindow {
    id: window
    width: 660
    height: 600
    minimumWidth: 520
    minimumHeight: 480
    visible: true
    title: "AI Music Detector"

    property bool busy: false
    property string verdict: ""
    property real artProb: -1
    property real sonProb: -1
    property string rawJson: ""
    property bool showDetails: false
    property string lang: bridge.language

    Material.theme: Material.Dark
    Material.accent: Material.Purple
    Material.primary: Material.BlueGrey

    function verdictColor(v) {
        if (v === "AI") return "#ff5252"
        if (v === "Human") return "#00e676"
        if (v === "Error") return "#ff9100"
        if (v === "Uncertain") return "#ffd740"
        return "#b0bec5"
    }

    function barColor(p) {
        if (p < 0) return "#546e7a"
        if (p >= 0.6) return "#ff5252"
        if (p >= 0.4) return "#ffd740"
        return "#00e676"
    }

    function toPath(url) {
        return decodeURIComponent(url.toString().replace(/^file:\/\//, ""))
    }

    component SignalBar: ColumnLayout {
        id: sb
        property string title: ""
        property real value: -1
        Layout.fillWidth: true
        spacing: 6

        RowLayout {
            Layout.fillWidth: true
            Label {
                text: sb.title
                font.bold: true
            }
            Item { Layout.fillWidth: true }
            Label {
                text: sb.value >= 0 ? (sb.value * 100).toFixed(1) + "%" : "n/a"
                color: window.barColor(sb.value)
                font.bold: true
            }
        }

        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 10
            radius: 5
            color: "#263238"

            Rectangle {
                width: parent.width * Math.max(0, Math.min(1, sb.value < 0 ? 0 : sb.value))
                height: parent.height
                radius: parent.radius
                color: window.barColor(sb.value)
                Behavior on width { NumberAnimation { duration: 350; easing.type: Easing.OutCubic } }
            }
        }
    }

    Connections {
        target: bridge

        function onResultChanged(text) {
            window.rawJson = text
            var o = null
            try { o = JSON.parse(text) } catch (e) { return }
            window.verdict = o.verdict || ""
            window.artProb = (o.p_ai_artifactnet !== undefined) ? o.p_ai_artifactnet : -1
            window.sonProb = (o.p_ai_sonics !== undefined) ? o.p_ai_sonics : -1
            window.showDetails = false
        }

        function onBusyChanged(b) {
            window.busy = b
        }
    }

    DropArea {
        anchors.fill: parent
        onDropped: function(drop) {
            if (drop.hasUrls && drop.urls.length > 0)
                bridge.detectFile(window.toPath(drop.urls[0]))
        }
    }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 20
        spacing: 14

        RowLayout {
            Layout.fillWidth: true

            ColumnLayout {
                spacing: 2
                Label {
                    text: "AI Music Detector"
                    font.pixelSize: 26
                    font.bold: true
                }
                Label {
                    text: bridge.tr(window.lang, "subtitle")
                    opacity: 0.7
                }
            }

            Item { Layout.fillWidth: true }

            ComboBox {
                id: langBox
                model: ["RU", "EN"]
                currentIndex: window.lang === "en" ? 1 : 0
                Layout.preferredWidth: 78
                onActivated: function(index) {
                    bridge.language = (index === 1 ? "en" : "ru")
                }
            }

            BusyIndicator {
                running: window.busy
                visible: running
            }
        }

        Rectangle {
            Layout.fillWidth: true
            visible: !bridge.sonicsAvailable
            radius: 8
            color: "#3e2723"
            border.width: 1
            border.color: "#ff9100"
            implicitHeight: sonicsWarn.implicitHeight + 20

            Label {
                id: sonicsWarn
                anchors.fill: parent
                anchors.margins: 10
                wrapMode: Text.WordWrap
                color: "#ffcc80"
                text: bridge.tr(window.lang, "sonics_warn")
            }
        }

        Rectangle {
            Layout.fillWidth: true
            Layout.fillHeight: true
            radius: 12
            color: "#1e272c"
            border.width: 2
            border.color: (window.verdict !== "") ? window.verdictColor(window.verdict) : "#37474f"

            ColumnLayout {
                anchors.fill: parent
                anchors.margins: 20
                spacing: 16

                RowLayout {
                    Layout.fillWidth: true
                    spacing: 10

                    Button {
                        text: bridge.tr(window.lang, "choose_file")
                        enabled: !window.busy
                        onClicked: bridge.chooseFile()
                    }
                    Button {
                        text: bridge.tr(window.lang, "record_mic")
                        enabled: !window.busy
                        highlighted: true
                        onClicked: bridge.detectMic()
                    }

                    Item { Layout.fillWidth: true }
                }

                Item {
                    Layout.fillWidth: true
                    Layout.fillHeight: true

                    ColumnLayout {
                        anchors.centerIn: parent
                        width: parent.width
                        spacing: 10

                        Label {
                            Layout.alignment: Qt.AlignHCenter
                            text: window.verdict === "" ? bridge.tr(window.lang, "drop_hint") : window.verdict
                            font.pixelSize: window.verdict === "" ? 20 : 40
                            font.bold: window.verdict !== ""
                            color: window.verdict === "" ? "#90a4ae" : window.verdictColor(window.verdict)
                        }

                        Label {
                            Layout.alignment: Qt.AlignHCenter
                            visible: window.verdict === "" && !window.busy
                            text: bridge.tr(window.lang, "formats_hint")
                            opacity: 0.6
                        }

                        Label {
                            Layout.alignment: Qt.AlignHCenter
                            visible: window.busy
                            text: bridge.tr(window.lang, "analyzing")
                            opacity: 0.8
                        }

                        ColumnLayout {
                            visible: window.verdict !== "" && window.verdict !== "Error"
                            Layout.fillWidth: true
                            Layout.topMargin: 12
                            Layout.leftMargin: 40
                            Layout.rightMargin: 40
                            spacing: 12

                            SignalBar {
                                title: bridge.tr(window.lang, "bar_artifactnet")
                                value: window.artProb
                            }
                            SignalBar {
                                title: bridge.tr(window.lang, "bar_sonics")
                                value: window.sonProb
                            }
                        }
                    }
                }
            }
        }

        RowLayout {
            Layout.fillWidth: true
            visible: window.rawJson !== ""

            Button {
                text: window.showDetails ? bridge.tr(window.lang, "hide_details")
                                         : bridge.tr(window.lang, "show_details")
                flat: true
                onClicked: window.showDetails = !window.showDetails
            }

            Item { Layout.fillWidth: true }
        }

        Flickable {
            Layout.fillWidth: true
            Layout.preferredHeight: window.showDetails ? 140 : 0
            visible: window.showDetails
            clip: true
            contentWidth: width
            contentHeight: detailText.paintedHeight

            TextEdit {
                id: detailText
                width: parent.width
                readOnly: true
                selectByMouse: true
                wrapMode: TextEdit.Wrap
                font.family: "monospace"
                font.pixelSize: 12
                color: "#b0bec5"
                text: window.rawJson
            }
        }
    }
}
