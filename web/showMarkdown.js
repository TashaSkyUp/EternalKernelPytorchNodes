import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";
import marked from "./marked.js";



// Displays input markdown on a node

app.registerExtension({
    name: "pysssss.ShowMarkdown",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
       if (nodeData.internal_state_display || nodeData.name == "ShowMarkdown") {
          // When the node is executed we will be sent the input markdown, display this in the widget
          const onExecuted = nodeType.prototype.onExecuted;
          nodeType.prototype.onExecuted = function (message) {
             onExecuted?.apply(this, arguments);

             if (this.widgets) {
                const pos = this.widgets.findIndex((w) => w.name === "markdown");
                if (pos !== -1) {
                   for (let i = pos; i < this.widgets.length; i++) {
                      this.widgets[i].onRemove?.();
                   }
                   this.widgets.length = pos;
                }
             }

             for (const list of message.text) {
                const w = ComfyWidgets["STRING"](this, "markdown", ["STRING", { multiline: true }], app).widget;

                // Convert markdown to HTML
                const html = marked(list);

                // Create a new div element to hold the rendered markdown
                const div = document.createElement("div");
                div.classList.add("markdown-widget");
                div.style.opacity = 0.6;
                div.innerHTML = html;

                // Replace the input element with the div containing the rendered markdown
                w.inputEl.parentNode.replaceChild(div, w.inputEl);
             }

             this.onResize?.(this.size);
          };
       }
    },
});