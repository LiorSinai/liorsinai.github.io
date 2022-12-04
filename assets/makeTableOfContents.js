document.addEventListener('DOMContentLoaded', function () {
    makeTableOfContents();
});


makeTableOfContents = () => {
    //var toc = document.createElement('nav');
    var toc = document.getElementById("toc");
    if (toc == null) {
        console.log("No element with id=toc found. Aborting table of contents creation.");
        return;
    }
    var list = document.createElement("ol")
    toc.appendChild(list)
    var content = document.getElementById("post-content");
    var headings = content.querySelectorAll('h1,h2,h3');

    for (i = 0; i <= headings.length - 1; i++) {
        var id = headings[i].innerHTML.toLowerCase().replace(/ /g, "-").replace(/[^a-z0-9-]/g, "");
        if (id == "table-of-contents") { continue; }
        var level = parseInt(headings[i].localName.replace("h", ""));
        var title = headings[i].innerHTML;
        var link = document.createElement('a');
        link.setAttribute("href", "#" + id);
        link.innerHTML = title;
        var item = document.createElement('li');
        item.appendChild(link);
        if (level == 2) {
            list.appendChild(item);
        }
        else if (level == 3) {
            if (list.children.length > 0) {
                var parent = list.children[list.children.length - 1]
            }
            else {
                var parent = document.createElement('li');
                list.appendChild(parent);
            }
            subheading = document.createElement('ul');
            subheading.appendChild(item);
            parent.appendChild(subheading)
        }
    }
}