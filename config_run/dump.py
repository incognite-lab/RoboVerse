def ik_solver_1(self) -> dict:
        import numpy as np
        import torch
        from ikpy.chain import Chain
        from scipy.spatial.transform import Rotation as R

        # =================================================================================================
        # 1. SETUP & LOGGING START
        # =================================================================================================
        print("\n" + "█" * 80)
        print("🛠️  IK SOLVER DEBUGGER START")
        print("█" * 80)

        env = self.env
        robot_cfg = env.scenario.robots[0]
        states = env.env.handler.get_states()
        robot_state = states.robots[robot_cfg.name]

        # Cesta k URDF (Ujisti se, že je správná)
        urdf_path = "/home/roboversepc/code/RoboVerse/roboverse_data/robots/g1/urdf/g1_rotslider_for_IK_left.urdf"

        # Base element = 'torso_link', aby rameno (shoulder joints) bylo součástí řešení!
        left_chain = Chain.from_urdf_file(urdf_path, base_elements=["torso_link"])
        print(f"✅ Chain Loaded. Active Links: {len(left_chain.links)}")

        # =================================================================================================
        # 2. TORSO BASE & OFFSET CORRECTION
        # =================================================================================================
        print("\n--- [2] BASE FRAME CALCULATION ---")
        torso_idx = robot_state.body_names.index("torso_link")

        # Raw pozice ze simulace (World Frame)
        torso_pos_raw = robot_state.body_state[0, torso_idx, :3].cpu().numpy()
        q_raw = robot_state.body_state[0, torso_idx, 3:7].cpu().numpy()

        print(f"  > Raw Sim Pos (COM):   {torso_pos_raw}")
        print(f"  > Raw Sim Quat:        {q_raw}")

        # !!! CRITICAL: QUATERNION ORDER CHECK !!!
        # MuJoCo vrací [w, x, y, z]. Scipy chce [x, y, z, w].
        # Pokud je robot rovně, w by mělo být 1 (nebo blízko).
        # Pokud je q_raw[0] (první prvek) cca 1.0, pak je to formát WXYZ a musíme to prohodit.
        if abs(q_raw[0]) > 0.9:
            # Je to WXYZ, převádíme na XYZW
            torso_quat = [q_raw[1], q_raw[2], q_raw[3], q_raw[0]]
            print("  > ⚠️ Quat format detected as WXYZ (MuJoCo standard). Swapped to XYZW for Scipy.")
        else:
            # Předpokládáme, že už to je XYZW (méně pravděpodobné u Simu)
            torso_quat = q_raw
            print("  > ℹ️ Quat format assumed XYZW.")

        rot_torso = R.from_quat(torso_quat)

        # !!! OFFSET CORRECTION (COM -> PIVOT) !!!
        # Hodnota z URDF <inertial><origin z="0.15082">
        # Musíme tento vektor odečíst v lokálním prostoru rotovaného těla.
        com_offset = np.array([0.000931, 0.000346, 0.15082])

        # Pivot = COM - (R * offset)
        offset_world = rot_torso.apply(com_offset)
        torso_pivot_pos = torso_pos_raw #- offset_world

        print(f"  > Offset Vector (URDF):{com_offset}")
        print(f"  > Offset in World:     {offset_world}")
        print(f"  > Corrected Pivot Pos: {torso_pivot_pos}  <-- TOTO JE SKUTEČNÁ BÁZE ŘETĚZCE")

        # Kontrola výšky: Pokud je pivot výš než COM, je něco špatně s rotací nebo znaménkem.
        if torso_pivot_pos[2] > torso_pos_raw[2]:
            print("  > 🔴 WARNING: Pivot je VYŠŠÍ než COM! To je divné (pas je pod hrudníkem). Zkontroluj Quaternion.")

        # Vytvoření matice Báze
        base_matrix = np.eye(4)
        base_matrix[:3, :3] = rot_torso.as_matrix()
        base_matrix[:3, 3] = torso_pivot_pos
        base_inv = np.linalg.inv(base_matrix)

        # =================================================================================================
        # 3. TARGET CALCULATION
        # =================================================================================================
        print("\n--- [3] TARGET CALCULATION ---")
        chair = states.objects["chair"]
        target_idx = chair.body_names.index('target_hand_left')

        target_pos_world = chair.body_state[0, target_idx, :3].cpu().numpy()
        q_tgt_raw = chair.body_state[0, target_idx, 3:7].cpu().numpy()

        # Opět kontrola/konverze quaternionu pro cíl
        target_quat = [q_tgt_raw[1], q_tgt_raw[2], q_tgt_raw[3], q_tgt_raw[0]] # Předpoklad WXYZ -> XYZW

        print(f"  > Target World Pos:    {target_pos_world}")

        # Skládání rotací
        r_chair = R.from_quat(target_quat)

        # FIX URDF: Endeffektor má v URDF roll -1.57. Musíme to vyrušit (+90 deg).
        r_urdf_fix = R.from_euler('z', -90, degrees=True)

        # APPROACH: Jak chceme chytit (otoč dle potřeby)
        r_approach = R.from_euler('y', -90, degrees=True)

        final_rot = r_chair * r_urdf_fix#* r_approach * r_urdf_fix

        target_matrix_world = np.eye(4)
        target_matrix_world[:3, :3] = final_rot.as_matrix()
        target_matrix_world[:3, 3] = target_pos_world

        # Transformace do Chain Frame
        target_in_chain = base_inv @ target_matrix_world
        print(f"  > Target Local (Chain):{target_in_chain[:3, 3]}")

        # Sanity check: Je cíl v dosahu? (G1 má ruku cca 0.7m dlouhou)
        dist_to_target = np.linalg.norm(target_in_chain[:3, 3])
        print(f"  > Distance to Target:  {dist_to_target:.4f} m")
        if dist_to_target > 0.8:
            print("  > 🔴 WARNING: Cíl je pravděpodobně mimo dosah (> 0.8m)!")

        # =================================================================================================
        # 4. SOLVE IK
        # =================================================================================================
        print("\n--- [4] SOLVING IK ---")
        ik_joints = left_chain.inverse_kinematics_frame(
            target_in_chain,
            orientation_mode="all",
            optimizer="least_squares"
        )
        print("  > IK Solution found.")

        # =================================================================================================
        # 5. MAPPING TO CONTROLLER (Link Name -> Joint Name)
        # =================================================================================================
        print("\n--- [5] MAPPING & DEBUG ---")
        full_joint_dict = {name: 0.0 for name in list(robot_cfg.joint_limits.keys())}

        # Header tabulky
        print(f"  {'Index':<5} | {'URDF Link Name':<30} | {'Calculated Angle':<10} | {'Status':<10} | {'Mapped To'}")
        print("  " + "-"*80)

        for i, link in enumerate(left_chain.links):
            link_name = link.name
            angle = ik_joints[i]

            # Logika pro nalezení správného jména jointu
            joint_name = None
            status = "❌ SKIP"

            # 1. Přímá shoda
            if link_name in full_joint_dict:
                joint_name = link_name
            # 2. Nahrazení _link za _joint (Nejčastější u G1)
            elif link_name.replace("_link", "_joint") in full_joint_dict:
                joint_name = link_name.replace("_link", "_joint")

            if joint_name:
                full_joint_dict[joint_name] = angle
                status = "✅ OK"
                print(f"  {i:<5} | {link_name:<30} | {angle: .4f}     | {status}    | {joint_name}")
            else:
                # Vypisujeme i ty přeskočené (často Fixed jointy, což je OK, ale musíme to vidět)
                if "fixed" not in link_name and "virtual" not in link_name and abs(angle) > 0.001:
                     print(f"  {i:<5} | {link_name:<30} | {angle: .4f}     | ⚠️ MISS   | ??? Check Names!")
                else:
                     # Fixed linky, které nás nezajímají
                     pass

        # =================================================================================================
        # 6. VERIFICATION (FK CHECK)
        # =================================================================================================
        print("\n--- [6] VERIFICATION (FK vs REALITY) ---")

        # A) FK (Teorie)
        fk_res = left_chain.forward_kinematics(ik_joints)
        fk_pos_local = fk_res[:3, 3]
        # Transformace z lokálu do světa pomocí NAŠÍ Base Matrix
        fk_pos_world_theory = (base_matrix @ np.append(fk_pos_local, 1.0))[:3]

        # B) Realita (Simulace)
        # Zde musíme vzít pozici endeffektoru, pokud existuje. Pokud robot stojí,
        # tato hodnota bude stará (před pohybem). Ale porovnáváme to s Targetem.
        try:
            real_ee_idx = robot_state.body_names.index("left_endeffector")
            real_ee_pos = robot_state.body_state[0, real_ee_idx, :3].cpu().numpy()
        except ValueError:
            real_ee_pos = np.array([0,0,0])

        # C) Porovnání FK Theory vs Target World (Měly by být shodné, pokud IK našlo řešení)
        ik_error = np.linalg.norm(fk_pos_world_theory - target_pos_world)

        print(f"  1. Target World:       {target_pos_world}")
        print(f"  2. IK Theory (FK):     {fk_pos_world_theory} (Kde by ruka MĚLA skončit)")
        print(f"  3. Sim Reality (Now):  {real_ee_pos} (Kde ruka je TEĎ - před pohybem)")

        print(f"\n  👉 IK Mathematical Error: {ik_error:.6f} m")

        if ik_error > 0.05:
            print("  🔴 FAIL: IK nenašlo řešení, které dosáhne na cíl! (Cíl je daleko nebo nemožná rotace)")
        else:
            print("  🟢 SUCCESS: IK matematicky sedí. Pokud se robot v simu netrefí, je chyba v PID/Simulaci/Offsetu.")

        print("█" * 80 + "\n")

        return full_joint_dict
