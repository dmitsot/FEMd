//  version.hpp -- the FEMd version.
//
//  This is the one place where the version is written. pyproject.toml reads
//  FEMD_VERSION from here for the Python package, and femd.__version__ comes
//  from the compiled module, so C++, the installed package and femd.__version__
//  always agree. Versions are X.Y (0.5, 0.6, ...), with X.Y.Z only for a fix
//  release of X.Y (0.5.1). To release a new version, write the changes under
//  "## Unreleased" in CHANGELOG.md and run ./bump_version.sh (minor by default,
//  or patch, major, X.Y, X.Y.Z), which updates these four lines, README.md, the
//  manual title and the CHANGELOG.
#ifndef FEMD_VERSION_HPP
#define FEMD_VERSION_HPP

#define FEMD_VERSION_MAJOR 0
#define FEMD_VERSION_MINOR 5
#define FEMD_VERSION_PATCH 0
#define FEMD_VERSION "0.5"

namespace femd {
inline constexpr const char* version = FEMD_VERSION;
}

#endif
