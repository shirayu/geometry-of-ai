<script setup lang="ts">
import { ref } from 'vue'

interface QuizData {
    choices: string[]
    answer: number
    explanation: string
    sources: string[]
}

const props = defineProps<{
    dataBase64: string
}>()

function decodeBase64Utf8(value: string): QuizData {
    const binary = atob(value)
    const bytes = Uint8Array.from(binary, (char) => char.charCodeAt(0))
    return JSON.parse(new TextDecoder().decode(bytes))
}

const quiz = decodeBase64Utf8(props.dataBase64)
const selected = ref<number | null>(null)

function select(i: number) {
    selected.value = i
}

function reset() {
    selected.value = null
}
</script>

<template>
    <div class="quiz">
        <ul class="quiz__choices">
            <li
                v-for="(choice, i) in quiz.choices"
                :key="i"
                class="quiz__choice-item"
            >
                <button
                    type="button"
                    class="quiz__choice"
                    :class="{
                        'quiz__choice--correct': selected !== null && i === quiz.answer,
                        'quiz__choice--wrong': selected === i && i !== quiz.answer,
                    }"
                    :aria-pressed="selected === i"
                    :disabled="selected !== null"
                    @click="select(i)"
                >
                    <span class="quiz__choice-num">{{ i + 1 }}</span>
                    <span class="quiz__choice-text" v-html="choice" />
                    <span class="quiz__choice-mark" aria-hidden="true">
                        <template v-if="selected !== null && i === quiz.answer">&#10003;</template>
                        <template v-else-if="selected === i && i !== quiz.answer">&#10007;</template>
                    </span>
                </button>
            </li>
        </ul>
        <Transition name="quiz-fade">
            <div
                v-if="selected !== null"
                class="quiz__result"
                :class="selected === quiz.answer ? 'quiz__result--correct' : 'quiz__result--wrong'"
                role="status"
                aria-live="polite"
            >
                <div class="quiz__result-head">
                    <p class="quiz__verdict">
                        <span v-if="selected === quiz.answer">正解</span>
                        <span v-else>不正解（正解は選択肢{{ quiz.answer + 1 }}）</span>
                    </p>
                    <button type="button" class="quiz__reset" @click="reset">もう一度</button>
                </div>
                <p class="quiz__explanation" v-html="quiz.explanation" />
            </div>
        </Transition>
        <p class="quiz__sources">
            本文の対応箇所：
            <a v-for="source in quiz.sources" :key="source" :href="source">参照する</a>
        </p>
    </div>
</template>

<style scoped>
.quiz {
    width: 100%;
    box-sizing: border-box;
    margin: 1rem 0 1.5rem;
    padding-left: 1.5rem;
    border-left: 2px solid var(--vp-c-divider);
}

.quiz__choices {
    list-style: none;
    margin: 0;
    padding: 0;
    display: flex;
    flex-direction: column;
    gap: 0.6rem;
}

.quiz__choice-item {
    margin: 0;
}

.quiz__choice {
    display: flex;
    align-items: center;
    width: 100%;
    box-sizing: border-box;
    gap: 0.7rem;
    padding: 0.7rem 1rem;
    border: 1.5px solid var(--vp-c-divider);
    border-radius: 8px;
    cursor: pointer;
    font: inherit;
    color: var(--vp-c-text-1);
    text-align: left;
    line-height: 1.5;
    transition: background-color 0.15s, border-color 0.15s;
}

.quiz__choice:disabled {
    cursor: default;
}

.quiz__choice:hover:not(.quiz__choice--correct):not(.quiz__choice--wrong) {
    border-color: var(--vp-c-brand-1);
    background: var(--vp-c-bg-mute);
}

.quiz__choice-num {
    flex: none;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 1.6em;
    height: 1.6em;
    border-radius: 50%;
    background: var(--vp-c-bg-mute);
    color: var(--vp-c-text-2);
    font-size: 0.85em;
    font-weight: 700;
}

.quiz__choice--correct .quiz__choice-num,
.quiz__choice--wrong .quiz__choice-num {
    background: transparent;
    color: inherit;
}

.quiz__choice-text {
    flex: 1;
}

.quiz__choice-mark {
    flex: none;
    width: 1.2em;
    text-align: center;
    font-weight: 700;
}

.quiz__choice--correct {
    border-color: var(--vp-c-green-1);
    background: var(--vp-c-green-soft);
    color: var(--vp-c-green-1);
    font-weight: 600;
}

.quiz__choice--correct .quiz__choice-text {
    color: var(--vp-c-text-1);
}

.quiz__choice--wrong {
    border-color: var(--vp-c-red-1);
    background: var(--vp-c-red-soft);
}

.quiz__choice--wrong .quiz__choice-mark {
    color: var(--vp-c-red-1);
}

.quiz__result {
    margin-top: 1.1rem;
    padding: 0.9rem 1.1rem;
    border-radius: 8px;
    border: 1px solid var(--vp-c-divider);
}

.quiz__result--correct {
    background: var(--vp-c-green-soft);
    border-color: var(--vp-c-green-1);
}

.quiz__result--wrong {
    background: var(--vp-c-red-soft);
    border-color: var(--vp-c-red-1);
}

.quiz__result-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.75rem;
    margin-bottom: 0.4rem;
}

.quiz__verdict {
    font-weight: 700;
    margin: 0;
}

.quiz__reset {
    flex: none;
    font-size: 0.8rem;
    font-weight: 600;
    padding: 0.3rem 0.7rem;
    border-radius: 999px;
    border: 1px solid var(--vp-c-divider);
    background: var(--vp-c-bg);
    color: var(--vp-c-text-2);
    cursor: pointer;
    transition: border-color 0.15s, color 0.15s;
}

.quiz__reset:hover {
    border-color: var(--vp-c-brand-1);
    color: var(--vp-c-brand-1);
}

.quiz__result--correct .quiz__verdict {
    color: var(--vp-c-green-1);
}

.quiz__result--wrong .quiz__verdict {
    color: var(--vp-c-red-1);
}

.quiz__explanation {
    margin: 0;
    color: var(--vp-c-text-1);
    font-size: 0.95em;
    line-height: 1.7;
}

.quiz__sources {
    margin: 0.75rem 0 0;
    color: var(--vp-c-text-2);
    font-size: 0.85em;
}

.quiz__sources a {
    margin-left: 0.4rem;
}

.quiz-fade-enter-active {
    transition: opacity 0.2s ease;
}

.quiz-fade-enter-from {
    opacity: 0;
}
</style>
