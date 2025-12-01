#!/usr/bin/env python3
"""
AddBiomechanics .b3d 파일에서 physics 데이터 추출
PhysPT fine-tuning을 위한 GT force, torque, acceleration 데이터 생성

Usage:
    python extract_addbiomechanics_physics.py \
        --b3d /path/to/subject.b3d \
        --out_dir /path/to/output \
        --processing_pass 2
"""

import nimblephysics as nimble
import numpy as np
import argparse
from pathlib import Path
import json


def extract_simple_physics_from_b3d(
    b3d_path: str,
    trial: int = 0,
    processing_pass: int = 2,
    start_frame: int = 0,
    num_frames: int = -1
) -> dict:
    """
    Extract simple physics data (GRF, CoM, vel, acc) from .b3d file

    This is a simplified version that doesn't require contact body information.

    Args:
        b3d_path: Path to .b3d file
        trial: Trial index
        processing_pass: Processing pass (0=KINEMATICS, 1=FILTER, 2=DYNAMICS)
        start_frame: Starting frame
        num_frames: Number of frames (-1 = all)

    Returns:
        dict with keys:
            - grf: (T, 3) total ground reaction force
            - com_pos: (T, 3) center of mass position
            - com_vel: (T, 3) center of mass velocity
            - com_acc: (T, 3) center of mass acceleration
            - vel: (T, N_dof) joint velocities
            - acc: (T, N_dof) joint accelerations
            - tau: (T, N_dof) joint torques
    """
    subject = nimble.biomechanics.SubjectOnDisk(b3d_path)

    # Get trial info
    trial_length = subject.getTrialLength(trial)
    if num_frames < 0:
        num_frames = trial_length - start_frame
    else:
        num_frames = min(num_frames, trial_length - start_frame)

    # Get skeleton to know DOF count
    skel = subject.readSkel(processingPass=processing_pass)
    n_dof = skel.getNumDofs()

    # Initialize arrays
    grf = np.zeros((num_frames, 3), dtype=np.float32)
    com_pos = np.zeros((num_frames, 3), dtype=np.float32)
    com_vel = np.zeros((num_frames, 3), dtype=np.float32)
    com_acc = np.zeros((num_frames, 3), dtype=np.float32)
    vel = np.zeros((num_frames, n_dof), dtype=np.float32)
    acc = np.zeros((num_frames, n_dof), dtype=np.float32)
    tau = np.zeros((num_frames, n_dof), dtype=np.float32)

    # Read frames
    frames = subject.readFrames(
        trial=trial,
        startFrame=start_frame,
        numFramesToRead=num_frames,
        includeProcessingPasses=True,
        includeSensorData=False,
        stride=1,
        contactThreshold=1.0
    )

    for i, frame in enumerate(frames):
        if len(frame.processingPasses) > processing_pass:
            pp = frame.processingPasses[processing_pass]

            # Ground reaction force - sum all forces
            grf_raw = np.array(pp.groundContactForce, dtype=np.float32)
            if len(grf_raw) > 0:
                grf_reshaped = grf_raw.reshape(-1, 3)
                grf[i] = grf_reshaped.sum(axis=0)  # Total GRF

            # Center of mass
            com_pos[i] = np.array(pp.comPos, dtype=np.float32)
            com_vel[i] = np.array(pp.comVel, dtype=np.float32)
            com_acc[i] = np.array(pp.comAcc, dtype=np.float32)

            # Joint velocities, accelerations, torques
            vel_raw = np.array(pp.vel, dtype=np.float32)
            acc_raw = np.array(pp.acc, dtype=np.float32)
            tau_raw = np.array(pp.tau, dtype=np.float32)

            vel[i] = vel_raw[:n_dof] if len(vel_raw) >= n_dof else np.pad(vel_raw, (0, n_dof - len(vel_raw)))
            acc[i] = acc_raw[:n_dof] if len(acc_raw) >= n_dof else np.pad(acc_raw, (0, n_dof - len(acc_raw)))
            tau[i] = tau_raw[:n_dof] if len(tau_raw) >= n_dof else np.pad(tau_raw, (0, n_dof - len(tau_raw)))

    return {
        'grf': grf,
        'com_pos': com_pos,
        'com_vel': com_vel,
        'com_acc': com_acc,
        'vel': vel,
        'acc': acc,
        'tau': tau,
        'n_dof': n_dof,
        'num_frames': num_frames
    }


def extract_physics_from_b3d(
    b3d_path: str,
    trial: int = 0,
    processing_pass: int = 2,  # DYNAMICS (가장 정확)
    start_frame: int = 0,
    num_frames: int = -1
) -> dict:
    """
    .b3d 파일에서 physics 데이터 추출

    Args:
        b3d_path: Path to .b3d file
        trial: Trial index (default: 0)
        processing_pass: Processing pass index (0=KINEMATICS, 1=LOW_PASS_FILTER, 2=DYNAMICS)
        start_frame: Start frame index
        num_frames: Number of frames to extract (-1 = all)

    Returns:
        dict with keys:
            - grf: (T, N_bodies, 3) ground reaction forces
            - cop: (T, N_bodies, 3) center of pressure
            - torques: (T, N_dof) joint torques
            - accelerations: (T, N_dof) joint accelerations
            - contacts: (T, N_bodies) binary contact states
            - residuals: (T, 6) root residual forces/moments
            - contact_body_names: (N_bodies,) body names
            - dof_names: (N_dof,) DOF names
            - metadata: dict with processing info
    """

    # Load .b3d file
    subject = nimble.biomechanics.SubjectOnDisk(b3d_path)

    # Get trial info
    trial_length = subject.getTrialLength(trial)
    if num_frames < 0:
        num_frames = trial_length - start_frame
    else:
        num_frames = min(num_frames, trial_length - start_frame)

    end_frame = start_frame + num_frames

    # Get skeleton info
    skel = subject.readSkel(processingPass=processing_pass)
    n_dof = skel.getNumDofs()
    n_bodies = skel.getNumBodyNodes()

    # Get body and DOF names
    contact_body_names = []
    contact_body_indices = []
    for i in range(n_bodies):
        body = skel.getBodyNode(i)
        # Check if body has shape nodes (collision geometry)
        num_shapes = body.getNumShapeNodes()
        if num_shapes > 0:
            contact_body_names.append(body.getName())
            contact_body_indices.append(i)

    n_contact_bodies = len(contact_body_names)

    dof_names = [skel.getDofByIndex(i).getName() for i in range(n_dof)]

    # Initialize arrays
    grf = np.zeros((num_frames, n_contact_bodies, 3), dtype=np.float32)
    cop = np.zeros((num_frames, n_contact_bodies, 3), dtype=np.float32)
    torques = np.zeros((num_frames, n_dof), dtype=np.float32)
    accelerations = np.zeros((num_frames, n_dof), dtype=np.float32)
    contacts = np.zeros((num_frames, n_contact_bodies), dtype=np.uint8)
    residuals = np.zeros((num_frames, 6), dtype=np.float32)

    # Extract frame by frame
    frames = subject.readFrames(
        trial=trial,
        startFrame=start_frame,
        numFramesToRead=num_frames,
        includeProcessingPasses=True,
        includeSensorData=False,
        stride=1,
        contactThreshold=1.0
    )

    for i, frame in enumerate(frames):
        # Extract from processingPass
        if hasattr(frame, 'processingPasses') and len(frame.processingPasses) > processing_pass:
            pp = frame.processingPasses[processing_pass]

            # Joint torques and accelerations (full DOF)
            tau_raw = np.array(pp.tau, dtype=np.float32)
            acc_raw = np.array(pp.acc, dtype=np.float32)

            # Handle size mismatch (trim or pad)
            if len(tau_raw) >= n_dof:
                torques[i] = tau_raw[:n_dof]
            else:
                torques[i, :len(tau_raw)] = tau_raw

            if len(acc_raw) >= n_dof:
                accelerations[i] = acc_raw[:n_dof]
            else:
                accelerations[i, :len(acc_raw)] = acc_raw

            # GRF, CoP, contacts (only contact bodies)
            frame_grf = np.array(pp.groundContactForce, dtype=np.float32).reshape(-1, 3)
            frame_cop = np.array(pp.groundContactCenterOfPressure, dtype=np.float32).reshape(-1, 3)
            frame_contact = np.array(pp.contact, dtype=np.uint8)

            # Filter to contact bodies only
            for j, body_idx in enumerate(contact_body_indices):
                if body_idx < len(frame_contact):
                    grf[i, j] = frame_grf[body_idx] if body_idx < len(frame_grf) else np.zeros(3)
                    cop[i, j] = frame_cop[body_idx] if body_idx < len(frame_cop) else np.zeros(3)
                    contacts[i, j] = frame_contact[body_idx]

            # Residuals
            if hasattr(pp, 'residuals'):
                res_raw = np.array(pp.residuals, dtype=np.float32)
                if len(res_raw) >= 6:
                    residuals[i] = res_raw[:6]

    # Compute statistics
    grf_norms = np.linalg.norm(grf, axis=2)
    mean_residual_force = float(np.linalg.norm(residuals[:, :3], axis=1).mean())
    mean_residual_moment = float(np.linalg.norm(residuals[:, 3:], axis=1).mean())
    max_grf = float(grf_norms.max()) if grf_norms.size > 0 else 0.0
    contact_ratio = float(contacts.sum() / contacts.size) if contacts.size > 0 else 0.0

    # Metadata
    metadata = {
        'b3d_path': b3d_path,
        'trial': trial,
        'processing_pass': processing_pass,
        'processing_pass_name': ['KINEMATICS', 'LOW_PASS_FILTER', 'DYNAMICS'][processing_pass],
        'start_frame': start_frame,
        'num_frames': num_frames,
        'n_dof': n_dof,
        'n_contact_bodies': n_contact_bodies,
        'mean_residual_force': mean_residual_force,
        'mean_residual_moment': mean_residual_moment,
        'max_grf': max_grf,
        'contact_ratio': contact_ratio
    }

    return {
        'grf': grf,
        'cop': cop,
        'torques': torques,
        'accelerations': accelerations,
        'contacts': contacts,
        'residuals': residuals,
        'contact_body_names': contact_body_names,
        'dof_names': dof_names,
        'metadata': metadata
    }


def main():
    parser = argparse.ArgumentParser(
        description='Extract physics data from AddBiomechanics .b3d file'
    )
    parser.add_argument('--b3d', required=True, help='Path to .b3d file')
    parser.add_argument('--out_dir', required=True, help='Output directory')
    parser.add_argument('--trial', type=int, default=0)
    parser.add_argument('--processing_pass', type=int, default=2,
                       help='0=KINEMATICS, 1=LOW_PASS_FILTER, 2=DYNAMICS (recommended)')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--num_frames', type=int, default=-1)

    args = parser.parse_args()

    print(f"Extracting physics from: {args.b3d}")
    print(f"Processing pass: {args.processing_pass} ({'KINEMATICS' if args.processing_pass == 0 else 'LOW_PASS_FILTER' if args.processing_pass == 1 else 'DYNAMICS'})")

    # Extract physics
    physics_data = extract_physics_from_b3d(
        args.b3d,
        trial=args.trial,
        processing_pass=args.processing_pass,
        start_frame=args.start,
        num_frames=args.num_frames
    )

    # Save to output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save physics data
    physics_path = out_dir / 'physics_data.npz'
    np.savez_compressed(
        physics_path,
        grf=physics_data['grf'],
        cop=physics_data['cop'],
        torques=physics_data['torques'],
        accelerations=physics_data['accelerations'],
        contacts=physics_data['contacts'],
        residuals=physics_data['residuals']
    )

    # Save metadata
    metadata = {
        'contact_body_names': physics_data['contact_body_names'],
        'dof_names': physics_data['dof_names'],
        'metadata': physics_data['metadata']
    }

    meta_path = out_dir / 'physics_metadata.json'
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✓ Physics data saved to: {physics_path}")
    print(f"✓ Metadata saved to: {meta_path}")
    print(f"\nStatistics:")
    print(f"  Frames: {physics_data['metadata']['num_frames']}")
    print(f"  DOFs: {physics_data['metadata']['n_dof']}")
    print(f"  Contact bodies: {physics_data['metadata']['n_contact_bodies']}")
    print(f"  Mean residual force: {physics_data['metadata']['mean_residual_force']:.2f} N")
    print(f"  Mean residual moment: {physics_data['metadata']['mean_residual_moment']:.2f} Nm")
    print(f"  Max GRF: {physics_data['metadata']['max_grf']:.2f} N")
    print(f"  Contact ratio: {physics_data['metadata']['contact_ratio']:.2%}")


if __name__ == '__main__':
    main()
