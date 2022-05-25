window.addEventListener('DOMContentLoaded', () => {

	const observer = new IntersectionObserver(entries => {
		entries.forEach(entry => {
			const id = entry.target.getAttribute('id');
			if (entry.intersectionRatio > 0) {
				document.querySelector(`#stickyTable li a[href="#${id}"]`).parentElement.classList.add('active');
			} else {
				document.querySelector(`#stickyTable li a[href="#${id}"]`).parentElement.classList.remove('active');
			}
		});
	});

	// Track all sections that have an `id` applied
	document.querySelectorAll("h2[id], h3[id]").forEach((section) => {
		let id = section['id'];
		if (document.querySelector(`#stickyTable li a[href="#${id}"]`) != null){
			observer.observe(section);
		}
	});
	
});