import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

app.registerExtension({
  name: "GDE.UI.RemoveDefaultButtons",
  async setup(app) {
    document.getElementById("comfy-load-default-button").remove();
    document.getElementById("comfy-clipspace-button").remove();
  },
});

app.registerExtension({
	name: "GDE.UI",
	async setup() {
		const menu = document.querySelector(".comfy-menu");
		const separator = document.createElement("hr");

		separator.style.margin = "20px 0";
		separator.style.width = "100%";
		menu.append(separator);

		// Save To Git Button
		const saveButton = document.createElement("button");
		saveButton.textContent = "Save To Git";
		saveButton.onclick = async () => {
			try {
				const json = JSON.stringify(app.graph.serialize(), null, 2);
				await api.fetchApi("/gde/git/save", {
					method: "POST",
					headers: {
						"Content-Type": "application/json",
					},
					body: JSON.stringify({ data: json }),
				});
			} catch (error) {
				console.error(error);
			}
		};
		menu.append(saveButton);

		// Load From Git Button
		const loadButton = document.createElement("button");
		loadButton.textContent = "Load From Git";
		loadButton.onclick = async () => {
			try {
				const response = await api.fetchApi("/gde/git/load", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                    }
                });
				const data = await response.json();
				app.loadGraphData(JSON.parse(data.data));
			} catch (error) {
				console.error(error);
			}
		};
		menu.append(loadButton);
	}
});