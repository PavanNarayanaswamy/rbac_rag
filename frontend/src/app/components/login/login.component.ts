import { Component, inject, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Router } from '@angular/router';

import { AuthService } from '../../services/auth.service';

@Component({
  selector: 'app-login',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './login.component.html',
  styleUrls: ['./login.component.css'],
})
export class LoginComponent {
  private auth = inject(AuthService);
  private router = inject(Router);

  username = '';
  password = '';
  loading = signal(false);
  error = signal<string | null>(null);

  // Demo accounts surfaced in the UI for quick testing.
  readonly demoAccounts = [
    { user: 'admin',  pass: 'admin123',  role: 'ADMIN'  },
    { user: 'ceo',    pass: 'ceo123',    role: 'CLevel' },
    { user: 'eng',    pass: 'eng123',    role: 'ENGG'   },
    { user: 'sales',  pass: 'sales123',  role: 'SALES'  },
    { user: 'hr',     pass: 'hr123',     role: 'HR'     },
    { user: 'intern', pass: 'intern123', role: 'INTERN' },
  ];

  fillDemo(user: string, pass: string) {
    this.username = user;
    this.password = pass;
  }

  submit() {
    if (!this.username || !this.password) {
      this.error.set('Username and password are required.');
      return;
    }
    this.error.set(null);
    this.loading.set(true);

    this.auth.login(this.username, this.password).subscribe({
      next: () => {
        this.loading.set(false);
        this.router.navigate(['/chat']);
      },
      error: (err) => {
        this.loading.set(false);
        this.error.set(
          err?.error?.detail ?? 'Login failed. Please check your credentials.'
        );
      },
    });
  }
}
