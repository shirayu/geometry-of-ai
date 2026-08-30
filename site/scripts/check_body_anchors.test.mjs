import { test } from 'node:test'
import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { slugifyHeading, checkBodyAnchorLinks, hasInlineMath, hasMathHeading } from './check_body_anchors.mjs'

test('濁点・半濁点付きの見出しは合成済み仮名のスラグになる', () => {
    // slugifyHeading は NFKD 分解後に U+0300-U+036F の結合文字を除去し、続いて NFC で
    // 結合濁点(U+3099)・結合半濁点(U+309A)を基底仮名と再合成する。そのため
    // 「ジ」「ダ」「ポ」等の濁点・半濁点付き仮名は合成済み（見た目そのまま）の形で
    // スラグに残る。quiz.md の S: リンクも合成済み文字列（コピペした見た目そのまま）
    // で書ける。site/.vitepress/config.mts のアンカーslugifyも同じ挙動に揃えてある。
    const slug = slugifyHeading('マージンを決定する「支柱」：サポートベクター')
    assert.equal(slug, 'マージンを決定する「支柱」-サポートベクター')
    assert.notEqual(slug, 'マージンを決定する「支柱」：サポートベクター'.normalize('NFKD'))
})

test('濁点を含まない見出しは通常の文字列と一致する', () => {
    const slug = slugifyHeading('多様体とは何か')
    assert.equal(slug, '多様体とは何か')
})

test('数字始まりの見出しには先頭にアンダースコアが付く', () => {
    const slug = slugifyHeading('2. 数式を恐れるな、しかし数式に溺れるな')
    assert.match(slug, /^_2-/)
})

test('本文内アンカーリンクは対応する見出しがあれば通る', () => {
    const file = path.join(os.tmpdir(), `check_body_anchors_test_${Date.now()}.md`)
    fs.writeFileSync(
        file,
        ['## 導入', '', '詳細は[統計学](#統計学の節)で扱う。', '', '## 統計学の節'].join('\n')
    )
    const errors = []
    checkBodyAnchorLinks(file, errors)
    fs.rmSync(file)
    assert.deepEqual(errors, [])
})

test('本文内アンカーリンクは対応する見出しがなければエラーになる', () => {
    const file = path.join(os.tmpdir(), `check_body_anchors_test_${Date.now()}.md`)
    fs.writeFileSync(file, ['## 導入', '', '詳細は[統計学](#存在しない見出し)で扱う。'].join('\n'))
    const errors = []
    checkBodyAnchorLinks(file, errors)
    fs.rmSync(file)
    assert.equal(errors.length, 1)
    assert.match(errors[0], /存在しません: #存在しない見出し/)
})

test('参考文献アンカー(#ref-*)は本文内アンカーリンクの検証対象外', () => {
    const file = path.join(os.tmpdir(), `check_body_anchors_test_${Date.now()}.md`)
    fs.writeFileSync(file, ['## 導入', '', '出典は[Amari, 1998](#ref-amari1998)。'].join('\n'))
    const errors = []
    checkBodyAnchorLinks(file, errors)
    fs.rmSync(file)
    assert.deepEqual(errors, [])
})

test('<a id>形式のアンカーもリンク先として認識される', () => {
    const file = path.join(os.tmpdir(), `check_body_anchors_test_${Date.now()}.md`)
    fs.writeFileSync(
        file,
        ['## 用語集', '', '<a id="kw-example"></a>説明文', '', '[用語](#kw-example)へ戻る'].join('\n')
    )
    const errors = []
    checkBodyAnchorLinks(file, errors)
    fs.rmSync(file)
    assert.deepEqual(errors, [])
})

test('hasInlineMathは$...$を含むテキストを検出する', () => {
    assert.equal(hasInlineMath('Levi-Civita接続（ $\\alpha = 0$ ）の位置づけ'), true)
    assert.equal(hasInlineMath('多様体とは何か'), false)
})

test('hasMathHeadingは数式を含む見出しを持つファイルを検出する', () => {
    const withMath = path.join(os.tmpdir(), `check_body_anchors_test_${Date.now()}_math.md`)
    fs.writeFileSync(withMath, ['## 導入', '', '### 集中度 $\\kappa$ の直感'].join('\n'))
    assert.equal(hasMathHeading(withMath), true)
    fs.rmSync(withMath)

    const withoutMath = path.join(os.tmpdir(), `check_body_anchors_test_${Date.now()}_plain.md`)
    fs.writeFileSync(withoutMath, ['## 導入', '', '### ふつうの見出し'].join('\n'))
    assert.equal(hasMathHeading(withoutMath), false)
    fs.rmSync(withoutMath)
})

test('数式見出しを含むファイルはアンカーリンク検証をスキップする（check_built_anchors.mjsに委任）', () => {
    const file = path.join(os.tmpdir(), `check_body_anchors_test_${Date.now()}.md`)
    fs.writeFileSync(
        file,
        [
            '## 導入',
            '',
            '### 集中度 $\\kappa$ の直感',
            '',
            '詳細は[存在しないはずのリンク](#存在しない見出し)を見よ。',
        ].join('\n')
    )
    const errors = []
    checkBodyAnchorLinks(file, errors)
    fs.rmSync(file)
    assert.deepEqual(errors, [])
})
