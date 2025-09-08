---
layout: default
title: "codeit AI 4기 4팀 초급 프로젝트"
description: "codeit AI 4기 4팀 초급 프로젝트 "
date: 2025-09-07
cache-control: no-cache
expires: 0
pragma: no-cache
---

## 🏥 코드잇 AI 엔지니어 4기 4팀 초급 프로젝트

### 📱 프로젝트 개요
**제목**: 모바일 애플리케이션으로 촬영한 약물 이미지에서 **최대 4개의 알약을 동시에 검출하고 분류**하여, 사용자에게 약물 정보 및 상호작용 경고를 제공하는 AI 시스템 개발

### 👥 팀원
- 김명환
- 김민혁
- 이건희
- 이현재
- 서동일

### 📅 프로젝트 기간
**2025년 9월 9일 ~ 2025년 9월 25일**

<hr style="margin: 30px 0;">

<script>

{% assign cur_dir = "/" %}
{% include cur_files.liquid %}

  var curDir = '{{- cur_file_dir -}}';
  var curFiles = {{- cur_files_json -}};
  var curPages = {{- cur_pages_json -}};
  
console.log('allFiles:', allFiles);
console.log('allPages:', allPages);

console.log('curDir:', curDir);
console.log('curFiles:', curFiles);
console.log('curPages:', curPages);

  var project_path = site.baseurl
  var site_url = `https://c0z0c.github.io${project_path}${curDir}`
  var raw_url = `https://raw.githubusercontent.com/c0z0c${project_path}/alpha${curDir}`;
  var git_url = `https://github.com/c0z0c${project_path}/blob/alpha${curDir}`
  var colab_url = `https://colab.research.google.com/github/c0z0c${project_path}/blob/alpha${curDir}`;
  
  console.log('site_url:', site_url);
  console.log('raw_url:', raw_url);
  console.log('git_url:', git_url);
  console.log('colab_url:', colab_url);

  curFiles.forEach(file => {
    if (!file.title) {
      file.title = file.name;
    }
  });

  const mdFiles = allPages.filter(page => 
    page.dir === '/md/' && page.name.endsWith('.md')
  );

  mdFiles.forEach(page => {
    // curFiles에 같은 name과 path가 있는지 확인
    const exists = curFiles.some(file => file.name === page.name && file.path === page.path);

    if (!exists) {
      // 확장자 추출
      let extname = '';
      if (page.name && page.name.includes('.')) {
        extname = '.' + page.name.split('.').pop();
      }

      // basename 추출
      let basename = page.name ? page.name.replace(new RegExp('\\.[^/.]+$'), '') : '';

      // modified_time 처리 (page.date가 없으면 빈 문자열)
      let modified_time = page.date || '';

      // curFiles 포맷에 맞게 변환해서 추가
      curFiles.push({
        name: page.name || '',
        path: page.path || '',
        extname: extname,
        modified_time: modified_time,
        basename: basename,
        url: page.url || '',
        title: page.title || basename  // 추가 필요
      });
    }
  });

console.log('mdFiles.length:', mdFiles.length);

console.log('curFiles.length:', curFiles.length);
console.log('curFiles:', curFiles);

</script>

<script>

  function getFileInfo(extname) {
    switch(extname.toLowerCase()) {
      case '.ipynb':
        return { icon: '📓', type: 'Jupyter Notebook' };
      case '.py':
        return { icon: '🐍', type: 'Python 파일' };
      case '.md':
        return { icon: '📝', type: 'Markdown 문서' };
      case '.json':
        return { icon: '⚙️', type: 'JSON 설정' };
      case '.zip':
        return { icon: '📦', type: '압축 파일' };
      case '.png':
      case '.jpg':
      case '.jpeg':
        return { icon: '🖼️', type: '이미지 파일' };
      case '.csv':
        return { icon: '📊', type: '데이터 파일' };
      case '.pdf':
        return { icon: '📄', type: 'PDF 문서' };
      case '.docx':
        return { icon: '📊', type: 'Word 문서' };
      default:
        return { icon: '📄', type: '파일' };
    }
  }

  // 파일 액션 버튼 생성 함수
  function getFileActions(file) {
    const fileName = file.name;
    const fileExt = file.extname.toLowerCase();
    const url = file.url.replace(/^\//, "");
    const path = file.path
    
    let actions = '';
    
    if (fileExt === '.md' && fileName !== 'index.md') {
      const mdName = fileName.replace('.md', '');
      actions += `<a href="${site_url}${url}" class="file-action" title="렌더링된 페이지 보기" target="_blank">🌐</a>`;
      actions += `<a href="${git_url}docs/${path}" class="file-action" title="GitHub에서 원본 보기" target="_blank">📖</a>`;
    } else if (fileExt === '.ipynb') {
      actions += `<a href="${git_url}${fileName}" class="file-action" title="GitHub에서 보기" target="_blank">📖</a>`;
      actions += `<a href="${colab_url}${fileName}" class="file-action" title="Colab에서 열기" target="_blank">🚀</a>`;
    } else if (fileExt === '.pdf') {
      actions += `<a href="${git_url}${fileName}" class="file-action" title="GitHub에서 보기" target="_blank">📖</a>`;
      actions += `<a href="https://docs.google.com/viewer?url=${raw_url}${fileName}" class="file-action" title="PDF 뷰어로 열기" target="_blank">📄</a>`;
    } else if (fileExt === '.docx') {
      actions += `<a href="${git_url}${fileName}" class="file-action" title="GitHub에서 보기" target="_blank">📖</a>`;
      actions += `<a href="https://docs.google.com/viewer?url=${raw_url}${fileName}" class="file-action" title="Google에서 열기" target="_blank">📊</a>`;
    } else if (fileExt === '.html') {
      actions += `<a href="${site_url}${fileName}" class="file-action" title="웹페이지로 보기" target="_blank">🌐</a>`;
      actions += `<a href="${git_url}${fileName}" class="file-action" title="GitHub에서 원본 보기" target="_blank">📖</a>`;
    } else {
      actions += `<a href="${git_url}${fileName}" class="file-action" title="파일 열기" target="_blank">📖</a>`;
    }
    
    return actions;
  }

  // DOM이 로드된 후 파일 목록 렌더링
  document.addEventListener('DOMContentLoaded', function() {
    const fileGrid = document.querySelector('.file-grid');
    
    if (curFiles.length === 0) {
      fileGrid.innerHTML = `
        <div class="empty-message">
          <span class="empty-icon">📄</span>
          <h3>파일이 없습니다</h3>
          <p>현재 이 위치에는 완료된 미션 파일이 없습니다.</p>
        </div>
      `;
      return;
    }

    let html = '';
    curFiles.forEach(file => 
    {
      if (file.name === 'index.md' || file.name === 'info.md') return;

      const fileInfo = getFileInfo(file.extname);
      const fileDate = file.modified_time ? new Date(file.modified_time).toLocaleDateString('ko-KR') : '';
      const actions = getFileActions(file);
      
      html += `
        <div class="file-item">
          <div class="file-icon">${fileInfo.icon}</div>
          <div class="file-info">
            <h4 class="file-name">${file.title}</h4>
            <p class="file-type">${fileInfo.type}</p>
            <p class="file-size">${fileDate}</p>
          </div>
          <div class="file-actions">
            ${actions}
          </div>
        </div>
      `;
    }
    );
    
    fileGrid.innerHTML = html;
  });
</script>

<h2>📖 프로젝트 문서 목록</h2>
<div class="file-grid">
  <!-- 파일 목록이 JavaScript로 동적 생성됩니다 -->
</div>

