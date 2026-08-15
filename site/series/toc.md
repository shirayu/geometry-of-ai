
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
    <div v-if="page.quizLink" class="toc-item toc-quiz">
      <a :href="page.quizLink">→ 理解度チェック</a>
    </div>
  </div>
</div>

<style scoped>
.toc-section {
  margin-bottom: 1.5rem;
  padding: 1rem 1.25rem;
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  background-color: var(--vp-c-bg);
}

.toc-section:nth-child(even) {
  background-color: var(--vp-c-bg-alt);
}

.toc-page-title {
  margin-top: 0;
  margin-bottom: 0.5rem;
  border-top: none;
  border-bottom: none;
  padding-top: 0;
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
  padding-left: 1.25rem;
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

.toc-quiz {
  margin-top: 0.35rem;
}

.toc-quiz a {
  font-size: 0.875em;
  color: var(--vp-c-text-3);
}
</style>
