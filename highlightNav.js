window.addEventListener('DOMContentLoaded', () => {
	const state = {};
	const observer = new IntersectionObserver(entries => {
		let activated = false;
		entries.forEach(entry => {
			const id = entry.target.getAttribute('id');
			if (entry.intersectionRatio > 0) {
				state[id] = true;
				activated = true;
			} else {
				state[id] = false;
			};
			if (activated){
				for (const id in state) {
					if (state[id]){
						document.querySelector(`.side-nav li a[href="#${id}"]`).parentElement.classList.add('active');
					}
					else {
						document.querySelector(`.side-nav li a[href="#${id}"]`).parentElement.classList.remove('active');
					};
				}
			}
		});
	});
	// Track all sections that have an `id` applied
	document.querySelectorAll("h2[id], h3[id]").forEach((section) => {
		let id = section['id'];
		if (document.querySelector(`.side-nav li a[href="#${id}"]`) != null){
			observer.observe(section);
		}
	});
});