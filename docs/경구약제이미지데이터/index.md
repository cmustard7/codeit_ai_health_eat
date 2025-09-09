---
layout: default
title: AI 모델 환경 설치가이드
description: AI 모델 환경 설치가이드
cache-control: no-cache
expires: 0
pragma: no-cache
author: "김명환"
---

# ✅ 발표자료

<script>

{%- assign cur_dir = "/경구약제이미지데이터/AI 모델 환경 설치가이드/" -%}
{%- include cur_files.liquid -%}

  var curDir = '{{- cur_file_dir -}}';
  var curFiles = {{- cur_files_json -}};
  var curPages = {{- cur_pages_json -}};

  var project_path = site.baseurl
  var site_url = `https://c0z0c.github.io${project_path}${curDir}`
  var raw_url = `https://raw.githubusercontent.com/c0z0c${project_path}/alpha${curDir}`;
  var git_url = `https://github.com/c0z0c${project_path}/blob/${branch}/docs${curDir}`
  var colab_url = `https://colab.research.google.com/github/c0z0c${project_path}/blob/alpha${curDir}`;

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
      actions += `<a href="${site_url}${url}" class="file-action" title="렌더링된 페이지 보기">🌐</a>`;
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

  // 하위 폴더 목록 파싱 함수
  function getSubDirectories(files, pages, currentDir) {
    const subDirs = new Set();
    
    // console.log('=== getSubDirectories Debug ===');
    // console.log('currentDir:', currentDir);
    
    // currentDir 정규화 (항상 /로 끝나도록)
    const normalizedCurrentDir = currentDir === '/' ? '/' : currentDir.endsWith('/') ? currentDir : currentDir + '/';
    
    // files에서 하위 폴더 추출
    files.forEach(file => {
      const filePath = file.path;
      // console.log('Processing file:', filePath);
      
      // 현재 디렉토리의 직접적인 하위 폴더만 찾기
      if (normalizedCurrentDir === '/') {
        // 루트 디렉토리인 경우
        if (filePath.startsWith('/') && filePath.indexOf('/', 1) > 0) {
          const firstSlashIndex = filePath.indexOf('/', 1);
          const subFolderName = filePath.substring(1, firstSlashIndex);
          const subDirPath = '/' + subFolderName + '/';
          // console.log('Found subfolder from file:', subDirPath);
          subDirs.add(subDirPath);
        }
      } else {
        // 하위 디렉토리인 경우
        if (filePath.startsWith(normalizedCurrentDir)) {
          const remainingPath = filePath.substring(normalizedCurrentDir.length);
          const slashIndex = remainingPath.indexOf('/');
          if (slashIndex > 0) {
            const subFolderName = remainingPath.substring(0, slashIndex);
            const subDirPath = normalizedCurrentDir + subFolderName + '/';
            // console.log('Found subfolder from file:', subDirPath);
            subDirs.add(subDirPath);
          }
        }
      }
    });

    // pages에서 하위 폴더 추출 (md 파일 제외하고 모든 페이지 처리)
    pages.forEach(page => {
      const pagePath = page.path;
      // console.log('Processing page:', pagePath);

      if (pagePath.startsWith('/md/')) return;
      if (pagePath.startsWith('/assets/')) return;
      
      // 현재 디렉토리의 직접적인 하위 폴더만 찾기
      if (normalizedCurrentDir === '/') {
        // 루트 디렉토리인 경우
        if (pagePath.startsWith('/') && pagePath.indexOf('/', 1) > 0) {
          const firstSlashIndex = pagePath.indexOf('/', 1);
          const subFolderName = pagePath.substring(1, firstSlashIndex);
          const subDirPath = '/' + subFolderName + '/';
          // console.log('Found subfolder from page:', subDirPath);
          subDirs.add(subDirPath);
        }
      } else {
        // 하위 디렉토리인 경우
        if (pagePath.startsWith(normalizedCurrentDir)) {
          const remainingPath = pagePath.substring(normalizedCurrentDir.length);
          const slashIndex = remainingPath.indexOf('/');
          if (slashIndex > 0) {
            const subFolderName = remainingPath.substring(0, slashIndex);
            const subDirPath = normalizedCurrentDir + subFolderName + '/';
            // console.log('Found subfolder from page:', subDirPath);
            subDirs.add(subDirPath);
          }
        }
      }
    });
    
    const result = Array.from(subDirs).sort();
    // console.log('Final subDirectories:', result);
    // console.log('=== End Debug ===');
    
    return result;
  }

  // 폴더 정보 가져오기 함수
  function getFolderInfo(folderPath) {
    const folderName = folderPath.split("/").filter(s => s).pop() || "root";
    
    // 폴더명에 따른 아이콘과 설명
    const folderMappings = {
      'md': { icon: '📝', desc: 'Markdown 문서' },
      '회의록': { icon: '📋', desc: '팀 회의록' },
      'assets': { icon: '🎨', desc: '정적 자원' },
      '경구약제이미지데이터': { icon: '💊', desc: '약물 데이터' },
      'AI 모델 환경 설치가이드': { icon: '⚙️', desc: '설치 가이드' },
      '경구약제 이미지 데이터(데이터 설명서, 경구약제 리스트)': { icon: '📊', desc: '데이터 설명서' },
      '발표자료': { icon: '📊', desc: '발표 자료' },
      '협업일지': { icon: '📓', desc: '협업 일지' }
    };
    
    return folderMappings[folderName] || { icon: '📁', desc: '폴더' };
  }

  // 폴더 액션 버튼 생성 함수
  function getFolderActions(folderPath) {
    const cleanPath = folderPath.replace(/\/$/, ''); // 끝의 / 제거
    return `
      <a href="${site_url}${cleanPath}/" class="file-action" title="폴더 열기">📖</a>
      <a href="${git_url}docs${cleanPath}/" class="file-action" title="GitHub에서 보기" target="_blank">📂</a>
    `;
  }

  // DOM이 로드된 후 파일 목록 렌더링
  document.addEventListener('DOMContentLoaded', function() {
    // 하위 폴더 목록 생성
    const allFilesData = allFiles;
    const allPagesData = allPages;
    const subDirectories = getSubDirectories(allFilesData, allPagesData, curDir);
    
    // console.log('subDirectories:', subDirectories);
    
    // 폴더 목록 렌더링
    const folderGrid = document.querySelector('.folder-grid');
    if (folderGrid) {
      if (subDirectories.length === 0) {
        folderGrid.innerHTML = `
          <div class="empty-message">
            <span class="empty-icon">📁</span>
            <h3>하위 폴더가 없습니다</h3>
            <p>현재 위치에는 하위 폴더가 없습니다.</p>
          </div>
        `;
      } else {
        let folderHtml = '';
        subDirectories.forEach(folderPath => {
          const folderInfo = getFolderInfo(folderPath);
          const folderName = folderPath.split("/").filter(s => s).pop() || "root";
          const actions = getFolderActions(folderPath);
          
          folderHtml += `
            <div class="file-item folder-item">
              <div class="file-icon">${folderInfo.icon}</div>
              <div class="file-info">
                <h4 class="file-name">${folderName}</h4>
                <p class="file-type">${folderInfo.desc}</p>
                <p class="file-size">폴더</p>
              </div>
              <div class="file-actions">
                ${actions}
              </div>
            </div>
          `;
        });
        
        folderGrid.innerHTML = folderHtml;
      }
    }

    // 파일 목록 렌더링
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

<h2>� 하위 폴더 목록</h2>
<div class="folder-grid">
  <!-- 폴더 목록이 JavaScript로 동적 생성됩니다 -->
</div>

<h2>�📖 프로젝트 문서 목록</h2>
<div class="file-grid">
  <!-- 파일 목록이 JavaScript로 동적 생성됩니다 -->
</div>
