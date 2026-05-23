
<script setup>
import { data } from './toc.data.ts'
</script>

# 全ページ目次

<div v-for="page in data" :key="page.file" :id="page.id" class="toc-section">
  <h2 class="toc-page-title">
    <a :href="page.link">{{ page.title }}</a>
  </h2>
  <div class="toc-headings">
    <div
      v-for="h in page.headings"
      :key="h.slug"
      :class="['toc-item', 'toc-h' + h.level]"
    >
      <a :href="page.link + '#' + h.slug">{{ h.text }}</a>
    </div>
  </div>
</div>

<style scoped>
.toc-section {
  margin-bottom: 2rem;
}

.toc-page-title {
  margin-top: 2rem;
  margin-bottom: 0.5rem;
  border-bottom: none;
  padding-bottom: 0;
}

.toc-page-title a {
  text-decoration: none;
}

.toc-page-title a:hover {
  text-decoration: underline;
}

.toc-headings {
  display: flex;
  flex-direction: column;
  gap: 0.15rem;
}

.toc-item a {
  text-decoration: none;
  font-weight: 400;
}

.toc-item a:hover {
  text-decoration: underline;
}

.toc-h3 {
  padding-left: 1.25rem;
}

.toc-h3 a {
  font-size: 0.875em;
  color: var(--vp-c-text-2);
}
</style>
