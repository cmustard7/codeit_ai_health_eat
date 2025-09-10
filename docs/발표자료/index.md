---
layout: default
title: 발표자료
description: 발표자료
cache-control: no-cache
expires: 0
pragma: no-cache
author: "김명환"
---

# ✅ 발표자료

<script>

{% assign cur_dir = "/발표자료/" %}
{% include cur_files.liquid %}
{% include page_values.html %}
{% include page_files.html %}

</script>

<div class="file-grid">
</div>

---

<div class="navigation-footer">
  <a href="{{- site.baseurl -}}/" class="nav-button home">
    <span class="nav-icon">🏠</span> 홈으로
  </a>
</div>