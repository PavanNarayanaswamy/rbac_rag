import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

import { AuthService } from './auth.service';

export interface SourceChunk {
  access_label: string;
  source: string;
  snippet: string;
  score?: number | null;
}

export interface QueryResponse {
  answer: string;
  sources: SourceChunk[];
  accessed_labels: string[];
}

@Injectable({ providedIn: 'root' })
export class ChatService {
  constructor(private http: HttpClient, private auth: AuthService) {}

  query(question: string, topK = 4): Observable<QueryResponse> {
    return this.http.post<QueryResponse>(this.auth.apiUrl('/api/query'), {
      question,
      top_k: topK,
    });
  }
}
