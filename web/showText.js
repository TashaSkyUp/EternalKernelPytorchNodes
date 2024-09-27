import {app} from "/scripts/app.js";
import {ComfyWidgets} from "/scripts/widgets.js";
//import {createAddNodeWidget} from "./AddNodeWidget.js";

app.registerExtension({
    name: "etk.ShowText",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        const use_name = "ETK Show Text";
        const use_name_2 = "ShowList";
        // create an array with these two
        const use_names =
            [
                "ETK Show Text",
                "ShowList",
                "ShowDict"
            ];
        // if nodeData.name in use_names
        if (use_names.includes(nodeData.name)) {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated?.apply(this, arguments);

                // const nodeParams = {
                //     buttonLabel: "Add Node",
                //     nodeName: "Show Text",
                //     outputType: "STRING",
                //     inputSlotName: "text"
                // };
                // createAddNodeWidget(this, app, nodeParams);

                if (!nodeData.widgets) {
                    return;
                } else {
                    const widget = this.widgets.find((w) => w.name === "text_display");
                    if (widget) {
                        widget.inputEl.readOnly = true;
                        widget.inputEl.style.opacity = 0.6;
                        widget.inputEl.style.userSelect = "text";
                        widget.inputEl.style.webkitUserSelect = "text";
                        widget.inputEl.style.msUserSelect = "text";
                        widget.inputEl.style.mozUserSelect = "text";
                    }
                }

            }
        }
        ;

        // Check if nodeData has a property 'internal_state_display' to identify nodes used for display.
        if (use_names.includes(nodeData.name)) {
            // Debugging: print the nodeData to the console.
            console.log(nodeData);

            // Function to populate the node with text widgets.
            function populate(text) {
                this.display_widget_idx = this.widgets.findIndex((w) => w.name === "text_display");
                // Remove the widget with the name of the text display widget
                const widget = this.widgets.find((w) => w.name === "text_display");
                if (widget) {
                    widget.onRemove?.();
                    this.widgets.splice(this.widgets.indexOf(widget), 1);
                }

                // Create a new widget with the name of the text display widget and the correct text
                const w = ComfyWidgets["STRING"](this, "text_display", ["STRING", {multiline: true}], app).widget;
                // Make the widget's input element read-only and semi-transparent.
                w.inputEl.readOnly = true;
                w.inputEl.style.opacity = 0.6;
                w.inputEl.style.userSelect = "text";
                w.inputEl.style.webkitUserSelect = "text";
                w.inputEl.style.msUserSelect = "text";
                w.inputEl.style.mozUserSelect = "text";

                // If text has only one item, update the w.value directly without indexing
                if (text.length === 1) {
                    w.value = text[0];
                } else {
                    // Set the value of the widget to the current text item, use the correct item index
                    w.value = text[this.display_widget_idx];
                }

                // Adjust the node size after adding widgets.
                requestAnimationFrame(() => {
                    const sz = this.computeSize();
                    // Ensure the node size is not smaller than its initial size.
                    if (sz[0] < this.size[0]) {
                        sz[0] = this.size[0];
                    }
                    if (sz[1] < this.size[1]) {
                        sz[1] = this.size[1];
                    }
                    // Trigger a resize event with the new size.
                    this.onResize?.(sz);
                    // Mark the canvas as dirty to update the UI.
                    app.graph.setDirtyCanvas(true, false);
                });
            }

            // Override the 'onExecuted' method to display text when the node is executed.
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                // Call the original 'onExecuted' method.
                onExecuted?.apply(this, arguments);
                // Populate the node with the text from the execution message.
                populate.call(this, message.text);
            };

            // Override the 'onConfigure' method to display stored widget values.
            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function () {
                // If there are stored widget values, populate the node with these values.
                if (this.widgets_values?.length) {
                    populate.call(this, this.widgets_values);
                }
            };
        }
    },
});