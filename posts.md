---
layout: page
title: "Posts"
permalink: /posts/
background:
---

Categories:
<ul>
{% for category in site.categories %}
  {% capture cat %}{{ category | first }}{% endcapture %}
  <li>
    <a href="#{{cat}}">{{ cat | capitalize }}</a>
  </li>
{% endfor %}
</ul>

---

{% for category in site.categories %}
  {% capture cat %}{{ category | first }}{% endcapture %}
  <h3 id="{{cat}}">{{ cat | capitalize }}</h3>
  <ul class="post-list">
  {% for post in site.categories[cat] %}
    <li>
	  <strong>
      <a href="{{ post.url | prepend: site.baseurl }}">{{ post.title }}</a>
	  </strong>
      <span class="post-date">- {{ post.date | date_to_long_string }}</span>
    </li>
  {% endfor %}
  </ul>
  {% if forloop.last == false %}<hr>{% endif %}
{% endfor %}
<br>
