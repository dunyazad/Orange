# Orange를 다른 프로젝트에서 라이브러리로 쓰기 (바이너리 배포)

Orange 엔진은 **소스를 복사하지 않고 바이너리 + 헤더**로 배포/소비할 수 있다.
`cmake --install` 이 만들어 주는 프리픽스가 곧 SDK이고, 소비자 프로젝트는
`find_package(Orange CONFIG REQUIRED)` 한 줄로 붙는다.

관련 파일:

| 파일 | 역할 |
|---|---|
| `cmake/OrangeInstall.cmake` | install/export 규칙 전부 (헤더·라이브러리·플러그인·패키지 설정) |
| `cmake/OrangeConfig.cmake.in` | 설치되는 `OrangeConfig.cmake` 템플릿 (`find_package` 진입점) |
| `cmake/OrangeRuntime.cmake` | 소비자용 헬퍼 `orange_copy_runtime()` / `orange_add_app()` |
| `scripts/package_sdk.ps1` | 빌드 + 설치를 한 번에 하는 패키징 스크립트 |
| `examples/consumer/` | SDK만 보고 빌드되는 예제 프로젝트 |

---

## 1. SDK 만들기 (배포하는 쪽)

```powershell
# Debug + Release 둘 다 -> dist/Orange-SDK-Debug, dist/Orange-SDK-Release
./scripts/package_sdk.ps1

# Release만, zip까지
./scripts/package_sdk.ps1 -Configs Release -Zip
```

수동으로 하면:

```powershell
cmake -S . -B build-sdk -G "Visual Studio 17 2022" -A x64 -DORANGE_INSTALL=ON
cmake --build   build-sdk --config Release
cmake --install build-sdk --config Release --prefix C:/SDK/Orange
```

### 설치 결과 레이아웃

```
<prefix>/
  include/orange/core/*.h              orange_core 공개 헤더
  include/orange/render/*.h            C ABI 렌더 계약 (플러그인 작성용)
  include/orange-third_party/
      robin_hood/robin_hood.h          sparse_grid.h 가 공개 헤더에서 include
      eigen/Eigen, eigen/unsupported   math.h 가 공개 헤더에서 include
  include/SDL3/…                       SDL3 (orange::core 의 public 의존성)
  lib/orange_core.lib                  정적 코어 (Debug 는 orange_cored.lib)
  lib/SDL3.lib
  lib/cmake/Orange/OrangeConfig.cmake  find_package 진입점
  lib/cmake/Orange/OrangeTargets*.cmake
  lib/cmake/Orange/OrangeRuntime.cmake
  lib/cmake/{SDL3,EnTT}/               같은 프리픽스로 재수출된 의존성
  bin/render_gl.dll, render_vk.dll     런타임에 dlopen 되는 렌더 백엔드
  bin/SDL3.dll (+ onnxruntime.dll)
```

핵심: **헤더 전용 의존성(robin_hood, Eigen)이 같이 들어간다.** 공개 헤더가
`<robin_hood/robin_hood.h>`, `<Eigen/Core>` 를 include 하므로, 이게 없으면
소비자가 별도로 Eigen을 깔아야 한다. 시스템 Eigen(`Eigen3::Eigen` 타깃)으로
빌드한 경우에만 번들을 생략하고, 그때는 `OrangeConfig.cmake` 가
`find_dependency(Eigen3)` 를 호출한다.

### 설정 옵션

| 옵션 | 기본값 | 의미 |
|---|---|---|
| `ORANGE_INSTALL` | `ON` | install/export 규칙 생성. `OFF` 면 예전처럼 빌드만 |
| `ORANGE_BUILD_SHARED` | `OFF` | `orange_core` 를 정적 `.lib` 대신 **DLL**로 빌드 |
| `ORANGE_BUILD_C_API` | `ON` | 순수 C 파사드 `orange_c.dll` 빌드 (아래 1-2절) |
| `ORANGE_STATIC_CRT` | `OFF` | MSVC 런타임 정적 링크(/MT). `orange_c.dll` 이 KERNEL32만 의존 |
| `ORANGE_INSTALL_APPS` | `OFF` | 데모 앱(`appOrange`, `cusp_probe`)도 `bin/` 에 설치 |

### DLL 배포 (`-Shared`)

```powershell
./scripts/package_sdk.ps1 -Shared -Configs Release
```

바뀌는 것은 `orange_core` 하나뿐:

```
<prefix>/bin/orange_core.dll     실제 코드 (~8 MB)
<prefix>/lib/orange_core.lib     import 라이브러리 (링크용, 심볼 테이블)
```

소비자 CMake 코드는 **정적일 때와 완전히 동일**하다 (`target_link_libraries(...
orange::core)`). `orange_copy_runtime()` 이 `bin/` 의 DLL을 전부 복사하므로
`orange_core.dll` 도 자동으로 exe 옆에 들어간다. 즉 배포 대상 프로젝트는
Orange 소스도, Orange 빌드도 필요 없고 `dist/Orange-SDK-Release` 폴더 하나만
받으면 된다.

헤더에 `__declspec(dllexport)` 주석이 하나도 없으므로 CMake의
`WINDOWS_EXPORT_ALL_SYMBOLS` 로 오브젝트 파일에서 `.def` 를 자동 생성한다
(내보낼 전역 **데이터** 심볼은 없다 — `DebugDraw` 는 함수 지역 static).
Unix에서는 `POSITION_INDEPENDENT_CODE` + `SOVERSION` 으로
`liborange_core.so.0` 가 나온다.

정적/DLL 선택 기준:

| | 정적 (`.lib`, 기본) | 공유 (`.dll`, `-Shared`) |
|---|---|---|
| 배포 파일 | exe 하나로 합쳐짐 | `orange_core.dll` 을 exe 옆에 둬야 함 |
| 링크 시간/exe 크기 | 큼 | 작음 |
| 엔진 업데이트 | 소비자 재링크 필요 | DLL만 교체 (시그니처가 그대로일 때) |
| 툴체인 제약 | 동일 MSVC + 런타임 | **동일** (C++ ABI라 완화되지 않음) |

`ORANGE_INSTALL=ON` 이면 `cmake/dependencies.cmake` 가 `SDL_INSTALL` /
`ENTT_INSTALL` 을 강제로 켠다. `orange_core` 가 SDL3/EnTT를 **public**으로
링크하므로 이 둘이 export 되지 않으면 `install(EXPORT OrangeTargets)` 자체가
실패한다. (`SDL_TEST_LIBRARY` 는 반대로 꺼둔다 — 빌드하지 않는 타깃의 설치
규칙이 남아 `cmake --install` 이 깨진다.)

---

## 1-2. C ABI (`orange_c.dll`) — 툴체인 제약 없는 길

`orange::core` 는 헤더에 STL/Eigen이 노출되는 C++ 라이브러리라 **소비자도 같은
MSVC로 빌드**해야 한다. 이 제약을 없애려고 CPU 툴킷 위에 순수 C 파사드를 얹었다.

- 헤더: `engine/c_api/include/orange/c/orange.h` (설치 후
  `<prefix>/include/orange/c/orange.h`) — `<stdint.h>` 외에 아무것도 include 안 함
- 바이너리: `<prefix>/bin/orange_c.dll` + `<prefix>/lib/orange_c.lib`
- 구현: `engine/c_api/src/orange_c.cpp`, `orange_core` 를 **PRIVATE** 링크하므로
  C++ 심볼이 밖으로 새지 않는다

```powershell
./scripts/package_sdk.ps1 -Configs Release -StaticCrt
```

`-StaticCrt`(= `-DORANGE_STATIC_CRT=ON`, /MT)를 주면 `orange_c.dll` 의 import가
**`KERNEL32.dll` 하나뿐**이다 (0.5 MB). VC++ 재배포 패키지도, SDL3.dll도,
`orange_core` 도 필요 없다 — **DLL 하나 + 헤더 하나**만 넘기면 끝.

```
orange_c.dll  0.49 MB   KERNEL32.dll 만 import (-StaticCrt)
orange_c.dll  0.20 MB   + MSVCP140/VCRUNTIME140 (기본 /MD)
```

### 제공하는 기능

| 그룹 | 함수 |
|---|---|
| 버전/오류 | `orangeVersion`, `orangeCApiVersion`, `orangeLastError` (스레드 로컬) |
| float3 배열 | `orangeVec3ArrayCreate/Count/Data/Copy/Destroy` |
| 포인트 IO | `orangePointsLoadFile` (ply/xyz/obj/off) |
| 포인트 연산 | `orangeEstimateNormals`, `orangeSmoothPoints`, `orangeIcpAlign` |
| 표면 복원 | `orangePointsToMesh`, `orangeReconstructFromFile` |
| 프리미티브 | `orangeBuildPlane/Box/Sphere/Cylinder/Cone/Torus/Disk/Capsule/Arrow` |
| 메시 접근 | `orangeMeshTriangleCount/Positions/Normals/Colors/Bounds/Destroy` |
| 공간 질의 | `orangeKdTreeCreate/Nearest/KNearest/Radius/Destroy` |

규약: 모든 함수 cdecl, 예외 없음(실패 시 `NULL` 또는 음수 `OrangeStatus` +
`orangeLastError()` 메시지), 핸들은 불투명 포인터이고 짝이 되는 `*Destroy` 로만
해제, 좌표는 xyz 연속 float, 메시 배열은 삼각형당 9 float, 행렬은 열 우선(4x4).

### CMake 소비자

```cmake
project(myTool LANGUAGES C)     # C로 충분하다
find_package(Orange CONFIG REQUIRED)
add_executable(myTool main.c)
target_link_libraries(myTool PRIVATE orange::c)
orange_copy_runtime(myTool)
```

### CMake 없이

```bat
cl /TC main.c /I C:\SDK\Orange\include C:\SDK\Orange\lib\orange_c.lib
copy C:\SDK\Orange\bin\orange_c.dll .
```

### 다른 언어 (예: Python ctypes)

```python
import ctypes
lib = ctypes.CDLL("orange_c.dll")
lib.orangeBuildSphere.restype  = ctypes.c_void_p
lib.orangeBuildSphere.argtypes = [ctypes.c_float, ctypes.c_int32, ctypes.POINTER(ctypes.c_float)]
mesh = lib.orangeBuildSphere(10.0, 48, None)
print(lib.orangeMeshTriangleCount(ctypes.c_void_p(mesh)))
lib.orangeMeshDestroy(ctypes.c_void_p(mesh))
```

예제: `examples/c_consumer/` (순수 C). 테스트: `engine/tests/test_c_api.c` →
CTest 타깃 `orange_c_tests` (래퍼 소스를 `ORANGE_C_STATIC` 으로 직접 링크해
DLL 로드 없이도 동일한 진입점을 검증).

`ORANGE_C_API_VERSION` 은 헤더가 하위 호환을 깨는 방식으로 바뀔 때 올린다.
로드 직후 `orangeCApiVersion()` 과 비교할 것.

---

## 2. SDK 쓰기 (소비하는 쪽)

```cmake
cmake_minimum_required(VERSION 3.21)
project(myApp LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

find_package(Orange CONFIG REQUIRED)

add_executable(myApp main.cpp)
target_link_libraries(myApp PRIVATE orange::core)
orange_copy_runtime(myApp)
```

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 `
      -DCMAKE_PREFIX_PATH=C:/SDK/Orange
cmake --build build --config Release
```

### 패키지가 제공하는 것

| 이름 | 종류 | 설명 |
|---|---|---|
| `orange::core` | 정적/공유 라이브러리 | ECS + 플랫폼 계층 + CPU 지오메트리/IO 툴킷. SDL3/EnTT/Eigen 인터페이스 포함 (C++ ABI) |
| `orange::c` | DLL | 순수 C 파사드. 툴체인/언어 무관 (1-2절) |
| `orange::render_api` | 헤더 전용 | 렌더 플러그인 C ABI. 자체 백엔드를 만들 때만 필요 |
| `ORANGE_PLUGIN_DIR` | 변수 | `render_gl`/`render_vk`/`SDL3.dll` 이 있는 디렉터리 |
| `ORANGE_INCLUDE_DIR` | 변수 | 설치된 헤더 루트 |
| `orange_copy_runtime(<target>)` | 함수 | 위 런타임 파일들을 타깃 exe 옆으로 POST_BUILD 복사 |
| `orange_add_app(<name> SOURCES …)` | 함수 | exe 생성 + `orange::core` 링크 + 런타임 복사를 한 번에 |

**`orange_copy_runtime()` 는 헤드리스 도구에도 필요하다.** `orange::core` 가
SDL3를 public으로 링크하므로 창을 안 띄워도 `SDL3.dll` 이 exe 옆에 있어야 한다.
렌더 플러그인은 링크 대상이 아니라 **실행 파일과 같은 디렉터리에서 런타임에
로드**되므로, 이 복사가 곧 플러그인 배치다.

### 두 가지 소비 시나리오

1. **툴킷만 쓰기 (창 없음)** — `orange/core/{kdtree,bvh,octree,sparse_grid,
   mesh_generation,primitives,point_ops,normals,serialization,modes}.h` 등 CPU
   유틸만 사용. 렌더러 플러그인은 로드되지 않는다. `examples/consumer/main.cpp`
   가 이 경우다.
2. **풀 앱** — `orange/core/application.h` + `window.h` 로 창/ECS 루프를 띄우고
   `render_gl`/`render_vk` 플러그인을 런타임에 로드. `orange_add_app()` 사용.

---

## 3. 주의사항

- **구성(Configuration)별 프리픽스.** 플러그인 DLL과 `SDL3.dll` 은 이름이
  구성과 무관해서, Debug와 Release를 한 프리픽스에 설치하면 나중 것이 앞의
  런타임을 덮는다. `package_sdk.ps1` 이 기본으로 `Orange-SDK-Debug` /
  `Orange-SDK-Release` 를 따로 만드는 이유다. 정적 라이브러리 자체는
  `orange_cored.lib`(Debug) / `orange_core.lib`(Release) 로 구분되므로 `.lib`
  만 필요하면 `-Single` 로 한 프리픽스에 합쳐도 된다.
- **MSVC ABI.** `orange_core` 의 인터페이스는 C++ 다 (STL/Eigen 타입이 헤더에
  그대로 노출). 정적이든 DLL이든 소비자는 **같은 툴체인·같은 런타임
  (/MD vs /MDd)** 으로 빌드해야 한다. 툴체인이 다르거나 다른 언어에서 쓸
  거라면 **`orange::c` (1-2절)** 를 쓰거나 소스로 받아
  `add_subdirectory(engine)` 하라. C ABI 경계는 `orange_c.dll` 과 렌더 플러그인
  (`orange/render/plugin_abi.h`) 두 곳뿐이다.
- **코드 서명 / Smart App Control.** Windows 11의 Smart App Control(레지스트리
  `HKLM:\SYSTEM\CurrentControlSet\Control\CI\Policy` 의
  `VerifiedAndReputablePolicyState = 1`)이 켜진 머신은 서명 없이 갓 빌드된
  DLL/EXE의 로드를 차단한다(`LoadLibrary` 오류 4551). 배포용 SDK라면
  `signtool` 로 `orange_core.dll` / `render_*.dll` / `SDL3.dll` 에 서명하는 것을
  권장한다. 개발 머신에서만 겪는 문제라면 빌드 산출물이 정책 예외 경로에
  있는지 확인할 것.
- **플러그인 ABI 버전.** 플러그인은 로드 시
  `ORANGE_PLUGIN_ABI_VERSION`(현재 **v18**, `orange/render/plugin_abi.h`)을
  검사한다. SDK의 `bin/` 플러그인과 헤더는 항상 같은 릴리스에서 가져온 것을
  써야 한다.
- **Vulkan.** `render_vk.dll` 은 빌드 머신에 Vulkan SDK가 있을 때만 패키지에
  들어간다. 없으면 GL 백엔드만 배포된다.
- **ONNX Runtime(MobileSAM).** 벤더링된 경우에만 `lib/onnxruntime.lib` +
  `bin/onnxruntime.dll` + `share/orange/models` 가 함께 설치되고,
  `OrangeConfig.cmake` 가 import 라이브러리를 다시 붙여 준다. 없으면 해당 기능은
  스텁이다.

---

## 4. 검증된 경로

이 문서의 절차는 다음으로 확인했다 (Windows 11, VS 2022, x64):

```
cmake -S . -B build-sdk -G "Visual Studio 17 2022" -A x64 -DORANGE_INSTALL=ON -DORANGE_BUILD_TESTS=OFF
cmake --build build-sdk --config Release --target orange_core render_gl
cmake --install build-sdk --config Release --prefix dist/Orange-SDK-Release
cmake -S examples/consumer -B build-consumer -G "Visual Studio 17 2022" -A x64 -DCMAKE_PREFIX_PATH=dist/Orange-SDK-Release
cmake --build build-consumer --config Release
build-consumer/Release/orange_consumer.exe
#   sphere: 9216 triangles / cloud: 27648 points / reconstructed: 15264 triangles
```

Debug 프리픽스(`orange_cored.lib` + `OrangeTargets-debug.cmake`)도 동일하게 확인.

C ABI(`orange_c`)는 CTest 타깃 `orange_c_tests` 로 전 항목 통과
(vec3 배열 왕복/짧은 버퍼 거부, 프리미티브 + 바운즈, 노멀 추정 + 복원 + 스무딩,
KD-트리 nearest/kNN/radius, ICP 이동량 복원 + 열 우선 레이아웃, NULL·없는 파일
오류 경로). 설치 결과에서 `orange_c.dll` 은 `-StaticCrt` 시 `KERNEL32.dll` 만,
기본(/MD) 시 MSVC 런타임만 import — SDL3/orange_core 의존 없음(`dumpbin
/dependents` 확인). 순수 C 예제(`examples/c_consumer`)도 SDK만으로 컴파일·링크
성공.

`-DORANGE_BUILD_SHARED=ON` 경로는 **빌드·설치·소비자 링크까지** 확인했다
(`bin/orange_core.dll` 8.3 MB + `lib/orange_core.lib` import 라이브러리,
`dumpbin /exports` 에 `buildSphere`/`pointsToMesh` 존재, 소비자 exe 링크 성공,
`orange_copy_runtime` 이 DLL을 exe 옆으로 복사). 다만 **실행 검증은 이 개발
머신에서 하지 못했다** — Smart App Control이 서명 없는 갓 빌드된 바이너리를
차단한다(위 주의사항 참고). 정적 구성은 같은 머신에서 실행까지 통과했다.
