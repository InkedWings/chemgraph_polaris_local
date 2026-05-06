# Main keywords and parameters obtained from https://wiki.fysik.dtu.dk/ase/_modules/ase/calculators/nwchem.html
# Parameters for NWChem calculator in CompChemAgent

import os
import sys
from pathlib import Path
from typing import Optional, Union, Dict
from pydantic import BaseModel, Field
from ase.calculators.nwchem import NWChem


class NWChemCalc(BaseModel):
    """NWChem quantum chemistry calculator configuration.

    This class defines the configuration parameters for NWChem quantum chemistry
    calculations. It supports various quantum chemical methods, basis sets, and
    periodic calculations through the NWChem program.

    Parameters
    ----------
    calculator_type : str, optional
        Calculator type. Currently supports only 'nwchem', by default 'nwchem'
    theory : str, optional
        NWChem module to be used. Options: 'dft', 'scf', 'mp2', 'ccsd', 'tce',
        'tddft', 'pspw', 'band', 'paw', by default 'dft'
    xc : str, optional
        Exchange-correlation functional (only applicable for DFT calculations),
        by default 'PBE'
    basis : str or dict, optional
        Basis set to use. Can be a string for all elements or a dictionary
        mapping elements to basis sets, by default '6-31G'
    kpts : tuple or dict, optional
        K-point mesh for periodic calculations, by default None
    directory : str, optional
        Working directory for NWChem calculations, by default '.'
    command : str, optional
        Command to execute NWChem (e.g., 'nwchem PREFIX.nwi > PREFIX.nwo'),
        by default None
    """

    calculator_type: str = Field(
        default="nwchem",
        description="Calculator type. Currently supports only 'nwchem'.",
    )
    theory: Optional[str] = Field(
        default="dft",
        description="NWChem module to be used. Options: 'dft', 'scf', 'mp2', 'ccsd', 'tce', 'tddft', 'pspw', 'band', 'paw'.",
    )
    xc: Optional[str] = Field(
        default="PBE",
        description="Exchange-correlation functional (only applicable for DFT calculations).",
    )
    basis: Optional[Union[str, Dict[str, str]]] = Field(
        default="6-31G",
        description="Basis set to use. Can be a string for all elements or a dictionary mapping elements to basis sets.",
    )
    kpts: Optional[Union[tuple, Dict[str, Union[int, str]]]] = Field(
        default=None, description="K-point mesh for periodic calculations."
    )
    directory: str = Field(
        default=".", description="Working directory for NWChem calculations."
    )
    command: Optional[str] = Field(
        default=None,
        description="Command to execute NWChem (e.g., 'nwchem PREFIX.nwi > PREFIX.nwo').",
    )

    @staticmethod
    def _ensure_nwchem_data_paths() -> None:
        """Set conda-forge NWChem data paths when conda activation was skipped."""
        prefix = Path(os.environ.get("CONDA_PREFIX", sys.prefix))
        share_dir = prefix / "share" / "nwchem"
        basis_dir = share_dir / "libraries"
        nwpw_dir = share_dir / "libraryps"

        if "NWCHEM_BASIS_LIBRARY" not in os.environ and basis_dir.is_dir():
            os.environ["NWCHEM_BASIS_LIBRARY"] = str(basis_dir) + os.sep

        if "NWCHEM_NWPW_LIBRARY" not in os.environ and nwpw_dir.is_dir():
            os.environ["NWCHEM_NWPW_LIBRARY"] = str(nwpw_dir) + os.sep

    @staticmethod
    def _default_work_directory() -> str:
        """Use the current ChemGraph session log dir for NWChem scratch files."""
        log_dir = os.environ.get("CHEMGRAPH_LOG_DIR")
        if not log_dir:
            return "."

        directory = Path(log_dir) / "nwchem"
        directory.mkdir(parents=True, exist_ok=True)
        return str(directory)

    def get_calculator(self):
        """Get an ASE-compatible NWChem calculator instance.

        Returns
        -------
        NWChem
            An ASE-compatible NWChem calculator instance

        Raises
        ------
        ValueError
            If an invalid calculator_type is specified
        """
        if self.calculator_type != "nwchem":
            raise ValueError(
                "Invalid calculator_type. The only valid option is 'nwchem'."
            )

        self._ensure_nwchem_data_paths()
        directory = self.directory
        if directory in ("", "."):
            directory = self._default_work_directory()

        return NWChem(
            theory=self.theory,
            xc=self.xc,
            basis=self.basis,
            kpts=self.kpts,
            directory=directory,
            command=self.command,
        )
