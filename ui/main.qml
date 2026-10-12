import QtQuick
import QtQuick.Controls
import QtQuick.Controls.Material
import QtQuick.Layouts

ApplicationWindow {
    id: window
    width: 680
    height: 660
    minimumWidth: 540
    minimumHeight: 520
    visible: true
    title: "AI Music Detector"
    color: window.colBg

    property bool busy: false
    property string verdict: ""
    property real artProb: -1
    property real sonProb: -1
    property real lofczProb: -1
    property var reasons: []
    property var warnings: []
    property var artWindows: []
    property var sonWindows: []
    property real progress: 0
    property string stageKey: "reading"
    property string rawJson: ""
    property bool showDetails: false
    property string lang: bridge.language
    property bool dark: bridge.darkTheme

    readonly property color colBg: dark ? "#12181c" : "#eceff1"
    readonly property color colCard: dark ? "#1e272c" : "#ffffff"
    readonly property color colText: dark ? "#eceff1" : "#263238"
    readonly property color colMuted: dark ? "#90a4ae" : "#607d8b"
    readonly property color colBorder: dark ? "#37474f" : "#cfd8dc"
    readonly property color colTrack: dark ? "#263238" : "#cfd8dc"

    Material.theme: dark ? Material.Dark : Material.Light
    Material.accent: Material.Purple
    Material.primary: Material.BlueGrey

    function verdictKey(v) {
        return "verdict_" + String(v).toLowerCase()
    }

    function verdictColor(v) {
        if (v === "AI") return "#ff5252"
        if (v === "Human") return "#00e676"
        if (v === "Error") return "#ff9100"
        if (v === "Uncertain") return "#ffd740"
        return "#b0bec5"
    }

    function barColor(p) {
        if (p < 0) return colTrack
        if (p >= 0.6) return "#ff5252"
        if (p >= 0.4) return "#ffd740"
        return "#00e676"
    }

    function reasonsText() {
        var parts = []
        for (var i = 0; i < reasons.length; i++)
            parts.push(bridge.tr(lang, "reason_" + reasons[i]))
        return parts.join(" · ")
    }

    function warningsText() {
        var parts = []
        for (var i = 0; i < warnings.length; i++)
            parts.push(bridge.tr(lang, "warn_" + warnings[i]))
        return parts.join(" · ")
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
                color: window.colText
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
            color: window.colTrack

            Rectangle {
                width: parent.width * Math.max(0, Math.min(1, sb.value < 0 ? 0 : sb.value))
                height: parent.height
                radius: parent.radius
                color: window.barColor(sb.value)
                Behavior on width { NumberAnimation { duration: 350; easing.type: Easing.OutCubic } }
            }
        }
    }

    component TimelineRow: ColumnLayout {
        id: tl
        property string title: ""
        property var values: []
        Layout.fillWidth: true
        spacing: 4

        Label {
            text: tl.title
            font.pixelSize: 11
            opacity: 0.8
            color: window.colMuted
        }

        Rectangle {
            id: track
            Layout.fillWidth: true
            Layout.preferredHeight: 14
            radius: 3
            color: window.colTrack
            clip: true

            Row {
                id: segRow
                anchors.fill: parent
                Repeater {
                    model: tl.values
                    Rectangle {
                        width: tl.values.length > 0 ? segRow.width / tl.values.length : 0
                        height: segRow.height
                        color: modelData === null ? window.colTrack : window.barColor(modelData)
                    }
                }
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
            window.lofczProb = (o.p_ai_lofcz !== undefined) ? o.p_ai_lofcz : -1
            window.reasons = o.reasons || []
            window.warnings = o.warnings || []
            window.artWindows = (o.windows && o.windows.artifactnet) ? o.windows.artifactnet : []
            window.sonWindows = (o.windows && o.windows.sonics) ? o.windows.sonics : []
            window.progress = 1
            window.showDetails = false
        }

        function onBusyChanged(b) {
            window.busy = b
            if (b) {
                window.progress = 0
                window.stageKey = "reading"
            }
        }

        function onProgressChanged(f, stage) {
            window.progress = f
            window.stageKey = stage
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
            spacing: 10

            ColumnLayout {
                spacing: 2
                Label {
                    text: "AI Music Detector"
                    font.pixelSize: 26
                    font.bold: true
                    color: window.colText
                }
                Label {
                    text: bridge.tr(window.lang, "subtitle")
                    opacity: 0.7
                    color: window.colText
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

            Button {
                id: themeBtn
                text: window.dark ? "\u263E" : "\u2600"
                flat: true
                onClicked: bridge.darkTheme = !window.dark
                ToolTip.visible: hovered
                ToolTip.text: window.dark ? bridge.tr(window.lang, "theme_light")
                                          : bridge.tr(window.lang, "theme_dark")
            }
        }

        RowLayout {
            Layout.fillWidth: true
            visible: window.busy
            spacing: 10

            BusyIndicator {
                running: window.busy && window.progress <= 0
                visible: running
                Layout.preferredWidth: 24
                Layout.preferredHeight: 24
            }

            ProgressBar {
                Layout.fillWidth: true
                visible: window.progress > 0
                value: window.progress
            }

            Label {
                text: bridge.tr(window.lang, "stage_" + window.stageKey)
                color: window.colMuted
                font.pixelSize: 12
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
            color: window.colCard
            border.width: 2
            border.color: (window.verdict !== "") ? window.verdictColor(window.verdict) : window.colBorder

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
                            text: window.verdict === ""
                                  ? bridge.tr(window.lang, "drop_hint")
                                  : bridge.tr(window.lang, window.verdictKey(window.verdict))
                            font.pixelSize: window.verdict === "" ? 20 : 40
                            font.bold: window.verdict !== ""
                            color: window.verdict === ""
                                   ? window.colMuted
                                   : window.verdictColor(window.verdict)
                        }

                        Label {
                            Layout.alignment: Qt.AlignHCenter
                            visible: window.verdict !== "" && reasonsText() !== ""
                            text: reasonsText()
                            color: window.colMuted
                            font.pixelSize: 13
                            horizontalAlignment: Text.AlignHCenter
                            wrapMode: Text.WordWrap
                            Layout.fillWidth: true
                        }

                        Label {
                            Layout.alignment: Qt.AlignHCenter
                            visible: window.verdict === "" && !window.busy
                            text: bridge.tr(window.lang, "formats_hint")
                            opacity: 0.6
                            color: window.colText
                        }

                        Label {
                            Layout.alignment: Qt.AlignHCenter
                            visible: window.busy
                            text: bridge.tr(window.lang, "analyzing")
                            opacity: 0.8
                            color: window.colText
                        }

                        Label {
                            Layout.alignment: Qt.AlignHCenter
                            visible: warnings.length > 0
                            text: warningsText()
                            color: "#ffab40"
                            font.pixelSize: 12
                            horizontalAlignment: Text.AlignHCenter
                            wrapMode: Text.WordWrap
                            Layout.fillWidth: true
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
                            SignalBar {
                                title: bridge.tr(window.lang, "bar_lofcz")
                                value: window.lofczProb
                            }
                        }

                        ColumnLayout {
                            visible: (window.artWindows.length > 1) || (window.sonWindows.length > 0)
                            Layout.fillWidth: true
                            Layout.topMargin: 8
                            Layout.leftMargin: 40
                            Layout.rightMargin: 40
                            spacing: 8

                            TimelineRow {
                                title: bridge.tr(window.lang, "timeline_artifactnet")
                                values: window.artWindows
                                visible: window.artWindows.length > 1
                            }
                            TimelineRow {
                                title: bridge.tr(window.lang, "timeline_sonics")
                                values: window.sonWindows
                                visible: window.sonWindows.length > 0
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
                color: window.colMuted
                text: window.rawJson
            }
        }
    }
}
