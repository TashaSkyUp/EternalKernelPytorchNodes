import {app} from "../../scripts/app.js";
import {api} from "../../scripts/api.js";

app.registerExtension({
    name: "movienodes.MovieMain",
    async init(app) {
        const canvas = document.querySelector("#graph-canvas");
        // Add event listener for the drop event
        canvas.addEventListener("drop", this.handleDrop.bind(this), false);

        //canvas.addEventListener("dragover", (e) => {
        //  e.preventDefault()
        //  console.log(e), false
        //});
        // Allow drop events on the app canvas
        canvas.addEventListener("dragover", this.handleDragOver.bind(this), false);
        canvas.addEventListener("dragenter", this.handleDragEnter.bind(this), false);
        canvas.addEventListener("dragleave", this.handleDragLeave.bind(this), false);
    },
    async handleDrop(event) {
        // Prevent default behavior to prevent the browser from opening the dropped file
        event.preventDefault();
        const supportsFileSystemAccessAPI =
            'getAsFileSystemHandle' in DataTransferItem.prototype;
        // Get the file(s) from the dropped event
        const files = event.dataTransfer.files;

        // Check if there are any files dropped
        if (files.length > 0) {
            // Iterate through the dropped files
            for (let i = 0; i < files.length; i++) {
                const fileHandlesPromises = [...event.dataTransfer.items]
                    // …by including only files (where file misleadingly means actual file _or_
                    // directory)…
                    //.filter((item) => item.kind === 'file')
                    // …and, depending on previous feature detection…
                    .map((item) =>
                        supportsFileSystemAccessAPI
                            // …either get a modern `FileSystemHandle`…
                            ? item.getAsFileSystemHandle()
                            // …or a classic `File`.
                            : item.getAsFile(),
                    );

                for await (const handle of fileHandlesPromises) {
                    // This is where we can actually exclusively act on the files.
                    console.log(`File: ${handle.name}`);
                }

                const file = fileHandlesPromises[i];
                const formData = new FormData();
                formData.append("name", file.name);
                formData.append("video", file);


                try {
                    const resp = await api.fetchApi("/SWAIN/SaveFile", {
                        method: "POST",
                        body: formData
                    });

                    if (resp.status === 200) {
                        // File uploaded successfully

                        // Get uploaded filename
                        const filename = await resp.json();

                        // Create node
                        const node = LiteGraph.createNode("VideoFileToImageStack");
                        app.graph.add(node);

                        // Set widget value to filename
                        // filter to the widget that has the name "vide_in"
                        const widget = node.widgets.find(w => w.name === "video_in");
                        widget.value =  file.name;



                    } else {
                        console.error("Upload failed");
                    }

                } catch (err) {
                    console.error("Error uploading file", err);
                }


            }
        }
    },
    handleDragOver(event) {
        // Allow dropping on the document
        event.preventDefault();
    }
    ,
    handleDragEnter(event) {
        // Add a class to the document when a file is dragged over it
        document.body.classList.add("dragover");
    }
    ,
    handleDragLeave(event) {
        // Remove the class from the document when a file is dragged out of it
        document.body.classList.remove("dragover");
    }
    ,
})
;