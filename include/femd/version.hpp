//  version.hpp -- the FEMd version.
//
//  This is the one place where the version is written. pyproject.toml reads
//  FEMD_VERSION from here for the Python package, and femd.__version__ comes
//  from the compiled module, so C++, the installed package and femd.__version__
//  always agree. To release a new version, write the changes under "## Unreleased"
//  in CHANGELOG.md and run ./bump_version.sh patch (or minor, major, X.Y.Z), which
//  updates these four lines, README.md, the manual title and the CHANGELOG.
#ifndef FEMD_VERSION_HPP
#define FEMD_VERSION_HPP

#define FEMD_VERSION_MAJOR 0
#define FEMD_VERSION_MINOR 1
#define FEMD_VERSION_PATCH 2
#define FEMD_VERSION "0.1.2"

namespace femd {
inline constexpr const char* version = FEMD_VERSION;
}

#endif
