document.addEventListener('DOMContentLoaded', function () {
    makeSideNav();
});


makeSideNav = () => {
    //var toc = document.createElement('nav');
    var toc = document.getElementById("stickyTable");
    if (toc == null) {
        console.log("No element with id=stickyTable found. Aborting side nav creation.");
        return;
    }
    var content = document.getElementById("post-content");
    var headings = content.querySelectorAll('h1,h2,h3');

    for (i = 0; i <= headings.length - 1; i++) {
        var id = headings[i].id;
        if (id == "table-of-contents") { continue; }
        var level = parseInt(headings[i].localName.replace("h", ""));
        var title = headings[i].innerHTML;
        var link = document.createElement('a'); 
        link.setAttribute("href", "#" + id);
        link.innerHTML = title;
        link.classList.add('nav-link');
        if (level == 2) {
            toc.appendChild(link);
        }
        else if (level == 3) {
            var idx = toc.children.length - 1;
            if ((idx > -1) && (toc.children[idx].nodeName = 'NAV')) {
                var parent = toc.children[idx]
            }
            else {
                var parent = document.createElement('nav');
                link.classList.add('nav');
                link.classList.add('flex-column');
                toc.appendChild(parent);
            }
            link.innerHTML = "-" + title;
            link.classList.add('ml-1');
            parent.appendChild(link)
        }
    }
}