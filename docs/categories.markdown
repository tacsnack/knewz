---
layout: page
title: Categories
permalink: /categories/
---

This is the base Jekyll theme. You can find out more info about customizing your Jekyll theme, as well as basic Jekyll usage documentation at [jekyllrb.com](https://jekyllrb.com/)

{% for tag in site.categories %}
# {{ tag | first }}
{% for post in tag[1] %}
- [{{ post.title }}]({{ site.baseurl }}{{ post.url }})
{% endfor %}
{% endfor %}